# marker_compiler.py - Add this to your Django app directory

import os
import subprocess
import tempfile
import json
from django.conf import settings
from pathlib import Path

class MindARCompiler:
    def __init__(self):
        self.node_modules_path = self.find_node_modules()
        
    def find_node_modules(self):
        """Find the node_modules directory"""
        # Check current directory first
        current_dir = Path.cwd()
        node_modules = current_dir / "node_modules"
        if node_modules.exists():
            return str(node_modules)
        
        # Check parent directories
        for parent in current_dir.parents:
            node_modules = parent / "node_modules"
            if node_modules.exists():
                return str(node_modules)
        
        return None
    
    def generate_marker_files(self, image_path, slug):
        """
        Generate MindAR marker files from an image using Node.js
        """
        try:
            if not self.node_modules_path:
                print("❌ node_modules not found. Please run 'npm install mind-ar' first.")
                return False
            
            # Create markers directory in static files
            static_dir = getattr(settings, 'STATICFILES_DIRS', ['static'])[0] if hasattr(settings, 'STATICFILES_DIRS') else 'static'
            marker_dir = os.path.join(static_dir, 'markers', slug)
            os.makedirs(marker_dir, exist_ok=True)
            
            # Create Node.js script for compilation
            node_script = self.create_compilation_script(image_path, marker_dir, slug)
            
            # Execute the Node.js script
            success = self.execute_node_script(node_script)
            
            if success:
                print(f"✅ Marker files generated successfully for {slug}")
                return True
            else:
                print(f"❌ Marker generation failed for {slug}")
                return False
                
        except Exception as e:
            print(f"❌ Error in marker generation: {e}")
            return False
    
    def create_compilation_script(self, image_path, output_dir, slug):
        """Create Node.js script for MindAR compilation"""
        
        script_content = f"""
const fs = require('fs');
const path = require('path');

// Import MindAR compiler
let Compiler;
try {{
    // Try different import methods
    Compiler = require('mind-ar/src/image-target/compiler').Compiler;
}} catch (e1) {{
    try {{
        const MindAR = require('mind-ar');
        Compiler = MindAR.Compiler;
    }} catch (e2) {{
        try {{
            Compiler = require('mind-ar').ImageCompiler;
        }} catch (e3) {{
            console.error('❌ Could not import MindAR Compiler:', e3.message);
            process.exit(1);
        }}
    }}
}}

async function compileMarker() {{
    try {{
        console.log('🎯 Starting marker compilation for: {slug}');
        console.log('📁 Input image: {image_path}');
        console.log('📁 Output directory: {output_dir}');
        
        // Check if input image exists
        if (!fs.existsSync('{image_path}')) {{
            throw new Error('Input image file not found: {image_path}');
        }}
        
        // Create output directory
        if (!fs.existsSync('{output_dir}')) {{
            fs.mkdirSync('{output_dir}', {{ recursive: true }});
        }}
        
        // Initialize compiler
        const compiler = new Compiler();
        
        // Read image file
        const imageBuffer = fs.readFileSync('{image_path}');
        
        // Compile the target image
        console.log('🔄 Compiling marker...');
        const result = await compiler.compile(imageBuffer);
        
        // Write marker files
        const files = {{
            '{slug}.iset': result.imageList || result.targetImages || '',
            '{slug}.fset': result.featurePoints || result.features || '',
            '{slug}.fset3': result.featurePoints3d || result.features3d || ''
        }};
        
        for (const [filename, content] of Object.entries(files)) {{
            const filepath = path.join('{output_dir}', filename);
            
            if (typeof content === 'string') {{
                fs.writeFileSync(filepath, content);
            }} else if (content instanceof Buffer) {{
                fs.writeFileSync(filepath, content);
            }} else if (typeof content === 'object') {{
                fs.writeFileSync(filepath, JSON.stringify(content));
            }} else {{
                // Create a minimal placeholder
                fs.writeFileSync(filepath, '# Generated marker file\\n');
            }}
            
            console.log(`✅ Created: ${{filename}}`);
        }}
        
        console.log('🎉 Marker compilation completed successfully!');
        process.exit(0);
        
    }} catch (error) {{
        console.error('❌ Compilation failed:', error.message);
        console.error(error.stack);
        process.exit(1);
    }}
}}

// Run the compilation
compileMarker();
"""
        return script_content
    
    def execute_node_script(self, script_content):
        """Execute the Node.js compilation script"""
        try:
            # Create temporary script file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(script_content)
                script_path = f.name
            
            # Execute with Node.js
            print("🔄 Executing Node.js compilation script...")
            result = subprocess.run(
                ['node', script_path], 
                capture_output=True, 
                text=True,
                cwd=os.path.dirname(self.node_modules_path) if self.node_modules_path else None
            )
            
            # Print output for debugging
            if result.stdout:
                print("📤 Node.js Output:", result.stdout)
            if result.stderr:
                print("⚠️ Node.js Errors:", result.stderr)
            
            # Clean up script file
            os.unlink(script_path)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Error executing Node.js script: {e}")
            return False
    
    def create_fallback_markers(self, marker_dir, slug):
        """Create basic marker files if compilation fails"""
        try:
            marker_files = {
                f'{slug}.iset': self.generate_basic_iset(slug),
                f'{slug}.fset': self.generate_basic_fset(slug),
                f'{slug}.fset3': self.generate_basic_fset3(slug)
            }
            
            for filename, content in marker_files.items():
                filepath = os.path.join(marker_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(content)
            
            print(f"✅ Created fallback marker files for {slug}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating fallback markers: {e}")
            return False
    
    def generate_basic_iset(self, slug):
        return f"""# MindAR Image Set File - {slug}
# Basic placeholder - replace with actual data
1
{slug}
1024 768
0
"""
    
    def generate_basic_fset(self, slug):
        return f"""# MindAR Feature Set File - {slug}
# Basic placeholder - replace with actual data
0
128
"""
    
    def generate_basic_fset3(self, slug):
        return f"""# MindAR 3D Feature Set File - {slug}
# Basic placeholder - replace with actual data
0
"""

# Usage in your Django views.py
def integrate_with_django_views():
    """
    Example integration with your existing Django views
    """
    example_code = '''
# Add this to your views.py imports:
from .marker_compiler import MindARCompiler

# Update your upload_view function:
def upload_view(request):
    if request.method == 'POST':
        # ... your existing code ...
        
        # After saving the experience
        if form.is_valid():
            experience = form.save()
            
            # Generate marker files
            compiler = MindARCompiler()
            marker_success = compiler.generate_marker_files(
                image_path=experience.image.path,
                slug=experience.slug
            )
            
            if not marker_success:
                # Fallback: create basic marker files
                static_dir = settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else 'static'
                marker_dir = os.path.join(static_dir, 'markers', experience.slug)
                os.makedirs(marker_dir, exist_ok=True)
                compiler.create_fallback_markers(marker_dir, experience.slug)
                
                print(f"⚠️ Using fallback markers for {experience.slug}")
            
            # ... rest of your code ...
    '''
    return example_code

# Test the compiler
def test_compiler():
    """Test function to verify the setup"""
    compiler = MindARCompiler()
    print(f"Node modules path: {compiler.node_modules_path}")
    
    if compiler.node_modules_path:
        print("✅ MindAR setup looks good!")
        return True
    else:
        print("❌ Please run 'npm install mind-ar' in your project directory")
        return False

if __name__ == "__main__":
    test_compiler()