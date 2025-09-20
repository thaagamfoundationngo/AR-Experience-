# ar_engine/ml_tracker.py
"""
Machine Learning-based AR tracker using PyTorch
Provides neural network pose estimation and advanced tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import cv2
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
import os

logger = logging.getLogger(__name__)

@dataclass
class MLTrackingResult:
    """Results from ML-based tracking"""
    detected: bool
    pose_6dof: Optional[np.ndarray]  # [tx, ty, tz, rx, ry, rz]
    confidence: float
    processing_time: float
    features: Optional[np.ndarray]
    homography: Optional[np.ndarray]

class PythonARTracker(nn.Module):
    """Enhanced PyTorch-based AR tracker with pose estimation"""
    
    def __init__(self, input_channels=3, feature_dim=128):
        super().__init__()
        
        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # Third conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Pose estimation head (6DOF: translation + rotation)
        self.pose_estimator = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 6)  # 6DOF pose
        )
        
        # Confidence estimation head
        self.confidence_estimator = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output 0-1 confidence
        )
        
        # Feature descriptor head
        self.descriptor_head = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim),
            nn.L2Norm(dim=1)  # Normalize features
        )
        
        # Model metadata
        self.input_size = (224, 224)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Training state
        self.training_history = []
        self.marker_templates = {}
        
        logger.info(f"Initialized ML tracker on {self.device}")
    
    def forward(self, x):
        """Forward pass through the network"""
        # Extract features
        features = self.feature_extractor(x)
        features_flat = features.view(features.size(0), -1)
        
        # Get pose, confidence, and descriptors
        pose = self.pose_estimator(features_flat)
        confidence = self.confidence_estimator(features_flat)
        descriptors = self.descriptor_head(features_flat)
        
        return {
            'pose': pose,
            'confidence': confidence,
            'descriptors': descriptors,
            'features': features
        }
    
    def train_on_marker(self, 
                       marker_images: List[np.ndarray], 
                       poses: List[np.ndarray],
                       marker_id: str = "default",
                       epochs: int = 100,
                       learning_rate: float = 0.001):
        """
        Train tracker on specific marker with known poses
        
        Args:
            marker_images: List of marker images from different viewpoints
            poses: Corresponding 6DOF poses [tx, ty, tz, rx, ry, rz]
            marker_id: Identifier for this marker
            epochs: Training epochs
            learning_rate: Learning rate
        """
        logger.info(f"Training ML tracker on marker '{marker_id}' with {len(marker_images)} samples")
        
        # Prepare data
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert to tensors
        images_tensor = []
        poses_tensor = []
        
        for img, pose in zip(marker_images, poses):
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            
            img_tensor = transform(img_rgb).unsqueeze(0)
            images_tensor.append(img_tensor)
            poses_tensor.append(torch.tensor(pose, dtype=torch.float32))
        
        images_batch = torch.cat(images_tensor, dim=0).to(self.device)
        poses_batch = torch.stack(poses_tensor).to(self.device)
        
        # Setup training
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        pose_criterion = nn.MSELoss()
        confidence_criterion = nn.BCELoss()
        
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self(images_batch)
            
            # Calculate losses
            pose_loss = pose_criterion(outputs['pose'], poses_batch)
            
            # Confidence loss (assume high confidence for training data)
            confidence_targets = torch.ones(len(images_batch), 1).to(self.device)
            confidence_loss = confidence_criterion(outputs['confidence'], confidence_targets)
            
            # Total loss
            total_loss = pose_loss + 0.1 * confidence_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Log progress
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {total_loss.item():.4f}")
        
        # Save marker template
        self.eval()
        with torch.no_grad():
            template_outputs = self(images_batch)
            template_descriptor = template_outputs['descriptors'].mean(dim=0)  # Average descriptor
        
        self.marker_templates[marker_id] = {
            'descriptor': template_descriptor.cpu().numpy(),
            'reference_pose': poses[0],  # Use first pose as reference
            'training_samples': len(marker_images),
            'training_timestamp': time.time()
        }
        
        logger.info(f"✅ Training completed for marker '{marker_id}'")
    
    def track_marker_ml(self, 
                       frame: np.ndarray, 
                       marker_id: str = "default") -> MLTrackingResult:
        """
        Track marker using ML model
        
        Args:
            frame: Input camera frame (BGR)
            marker_id: ID of trained marker to track
            
        Returns:
            MLTrackingResult with pose and confidence
        """
        start_time = time.time()
        
        if marker_id not in self.marker_templates:
            logger.warning(f"Marker '{marker_id}' not trained")
            return MLTrackingResult(False, None, 0.0, time.time() - start_time, None, None)
        
        try:
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(frame_rgb).unsqueeze(0).to(self.device)
            
            # Inference
            self.eval()
            with torch.no_grad():
                outputs = self(input_tensor)
                
                pose = outputs['pose'].cpu().numpy()[0]
                confidence = outputs['confidence'].cpu().numpy()[0, 0]
                descriptors = outputs['descriptors'].cpu().numpy()[0]
            
            # Check similarity with template
            template = self.marker_templates[marker_id]
            template_desc = template['descriptor']
            
            # Calculate descriptor similarity
            similarity = np.dot(descriptors, template_desc)
            similarity = (similarity + 1) / 2  # Normalize to 0-1
            
            # Combine ML confidence with descriptor similarity
            final_confidence = confidence * similarity
            
            # Threshold for detection
            detection_threshold = 0.5
            detected = final_confidence > detection_threshold
            
            processing_time = time.time() - start_time
            
            # Convert pose to homography for compatibility
            homography = None
            if detected:
                homography = self._pose_to_homography(pose, frame.shape)
            
            return MLTrackingResult(
                detected=detected,
                pose_6dof=pose if detected else None,
                confidence=final_confidence,
                processing_time=processing_time,
                features=descriptors,
                homography=homography
            )
            
        except Exception as e:
            logger.error(f"ML tracking error: {str(e)}")
            return MLTrackingResult(False, None, 0.0, time.time() - start_time, None, None)
    
    def _pose_to_homography(self, pose: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Convert 6DOF pose to homography matrix for visualization"""
        try:
            # Extract translation and rotation
            tx, ty, tz, rx, ry, rz = pose
            
            # Create rotation matrix from Euler angles
            R_x = np.array([[1, 0, 0],
                           [0, np.cos(rx), -np.sin(rx)],
                           [0, np.sin(rx), np.cos(rx)]])
            
            R_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                           [0, 1, 0],
                           [-np.sin(ry), 0, np.cos(ry)]])
            
            R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                           [np.sin(rz), np.cos(rz), 0],
                           [0, 0, 1]])
            
            R = R_z @ R_y @ R_x
            
            # Simple camera matrix (assuming typical webcam)
            fx = fy = frame_shape[1] * 0.8  # Approximate focal length
            cx, cy = frame_shape[1] / 2, frame_shape[0] / 2
            
            K = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])
            
            # Project 3D points to 2D
            object_points = np.array([[-0.1, -0.1, 0],
                                     [0.1, -0.1, 0],
                                     [0.1, 0.1, 0],
                                     [-0.1, 0.1, 0]], dtype=np.float32)
            
            # Transform points
            t = np.array([tx, ty, tz]).reshape(3, 1)
            transformed_points = (R @ object_points.T + t).T
            
            # Project to image plane
            projected = []
            for point in transformed_points:
                if point[2] > 0:  # In front of camera
                    x = (point[0] / point[2]) * fx + cx
                    y = (point[1] / point[2]) * fy + cy
                    projected.append([x, y])
                else:
                    projected.append([cx, cy])  # Default fallback
            
            # Create homography from projected points
            dst_points = np.array(projected, dtype=np.float32)
            src_points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
            
            homography = cv2.getPerspectiveTransform(src_points, dst_points)
            return homography
            
        except Exception as e:
            logger.error(f"Pose to homography conversion error: {str(e)}")
            return np.eye(3)
    
    def save_model(self, filepath: str, marker_id: str = None):
        """Save trained model and templates"""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'marker_templates': self.marker_templates,
            'input_size': self.input_size,
            'training_history': self.training_history
        }
        
        if marker_id and marker_id in self.marker_templates:
            save_dict['marker_templates'] = {marker_id: self.marker_templates[marker_id]}
        
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and templates"""
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.load_state_dict(checkpoint['model_state_dict'])
            self.marker_templates = checkpoint.get('marker_templates', {})
            self.input_size = checkpoint.get('input_size', (224, 224))
            self.training_history = checkpoint.get('training_history', [])
            
            logger.info(f"Model loaded from {filepath}")
            logger.info(f"Loaded {len(self.marker_templates)} marker templates")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def generate_training_data(self, 
                              marker_image_path: str, 
                              num_samples: int = 100) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate synthetic training data from a single marker image
        
        Args:
            marker_image_path: Path to marker image
            num_samples: Number of synthetic samples to generate
            
        Returns:
            Tuple of (images, poses)
        """
        base_image = cv2.imread(marker_image_path)
        if base_image is None:
            raise ValueError(f"Could not load marker image: {marker_image_path}")
        
        images = []
        poses = []
        
        for i in range(num_samples):
            # Generate random pose variations
            tx = np.random.uniform(-0.2, 0.2)
            ty = np.random.uniform(-0.2, 0.2)
            tz = np.random.uniform(0.5, 2.0)
            rx = np.random.uniform(-0.3, 0.3)
            ry = np.random.uniform(-0.3, 0.3)
            rz = np.random.uniform(-0.2, 0.2)
            
            pose = np.array([tx, ty, tz, rx, ry, rz])
            
            # Apply random transformations to image
            transformed_img = self._apply_synthetic_transform(base_image, pose)
            
            images.append(transformed_img)
            poses.append(pose)
        
        logger.info(f"Generated {num_samples} synthetic training samples")
        return images, poses
    
    def _apply_synthetic_transform(self, image: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Apply synthetic transformations to simulate pose changes"""
        h, w = image.shape[:2]
        
        # Apply rotation
        angle = pose[5] * 180 / np.pi  # Convert rz to degrees
        M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        
        # Apply translation
        M_rot[0, 2] += pose[0] * 100  # Scale translation
        M_rot[1, 2] += pose[1] * 100
        
        # Apply scale based on tz
        scale = 1.0 / pose[2]
        M_scale = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
        
        # Combine transformations
        transformed = cv2.warpAffine(image, M_rot, (w, h))
        transformed = cv2.warpAffine(transformed, M_scale, (w, h))
        
        # Add noise and lighting variations
        transformed = self._add_synthetic_variations(transformed)
        
        return transformed
    
    def _add_synthetic_variations(self, image: np.ndarray) -> np.ndarray:
        """Add realistic variations for training"""
        # Brightness variation
        brightness = np.random.uniform(0.7, 1.3)
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        # Contrast variation
        contrast = np.random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
        
        # Gaussian noise
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        
        # Gaussian blur
        if np.random.random() > 0.5:
            kernel_size = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return image

class L2Norm(nn.Module):
    """L2 normalization layer"""
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=self.dim)
