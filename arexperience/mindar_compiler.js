// mindar_compiler.js
const fs = require('fs');
const path = require('path');

async function compileTarget() {
    try {
        // Get paths from command line arguments
        const imagePath = process.argv[2];
        const targetPath = process.argv[3];
        
        if (!imagePath || !targetPath) {
            throw new Error('Missing required arguments: image path and target path');
        }
        
        console.log('Loading image:', imagePath);
        
        // Verify image file exists
        if (!fs.existsSync(imagePath)) {
            throw new Error('Image file does not exist: ' + imagePath);
        }
        
        // Read image file
        const imageBuffer = fs.readFileSync(imagePath);
        
        // Try multiple possible paths for the MindAR compiler
        const mindarPath = path.join(process.cwd(), 'node_modules', 'mind-ar');
        const possibleCompilerPaths = [
            path.join(mindarPath, 'dist', 'image-target', 'compiler.js'),
            path.join(mindarPath, 'src', 'image-target', 'compiler.js'),
            path.join(mindarPath, 'image-target', 'compiler.js'),
            path.join(mindarPath, 'dist', 'compiler.js'),
            path.join(mindarPath, 'compiler.js')
        ];
        
        let compilerPath = null;
        for (const possiblePath of possibleCompilerPaths) {
            if (fs.existsSync(possiblePath)) {
                compilerPath = possiblePath;
                console.log('Found MindAR compiler at:', compilerPath);
                break;
            }
        }
        
        if (!compilerPath) {
            throw new Error('MindAR compiler not found. Checked paths:\n' + possibleCompilerPaths.join('\n'));
        }
        
        // Convert path to file:// URL for Windows
        const fileUrl = pathToFileURL(compilerPath).href;
        console.log('Loading compiler from:', fileUrl);
        
        // Import the compiler using dynamic import
        const compilerModule = await import(fileUrl);
        
        // Get the Compiler class - handle different export formats
        let Compiler;
        if (compilerModule.Compiler) {
            Compiler = compilerModule.Compiler;
        } else if (compilerModule.default && compilerModule.default.Compiler) {
            Compiler = compilerModule.default.Compiler;
        } else if (compilerModule.default) {
            Compiler = compilerModule.default;
        } else {
            throw new Error('Could not find Compiler class in mind-ar module');
        }
        
        if (typeof Compiler !== 'function') {
            throw new Error('Compiler is not a function');
        }
        
        // Create compiler instance
        const compiler = new Compiler();
        
        // Compile the image target
        const compiledData = await compiler.compileImageTargets([{
            image: imageBuffer,
            name: 'target'
        }]);
        
        // Ensure target directory exists
        const targetDir = path.dirname(targetPath);
        if (!fs.existsSync(targetDir)) {
            fs.mkdirSync(targetDir, { recursive: true });
        }
        
        // Save the compiled target
        fs.writeFileSync(targetPath, JSON.stringify(compiledData));
        
        console.log('MindAR target compiled successfully:', targetPath);
        process.exit(0);
        
    } catch (error) {
        console.error('MindAR compilation failed:', error);
        process.exit(1);
    }
}

// Helper function to convert path to file:// URL
function pathToFileURL(path) {
    // Normalize path separators
    const normalizedPath = path.replace(/\\/g, '/');
    
    // Add drive letter and convert to file:// URL
    if (normalizedPath[1] === ':' && normalizedPath[0].match(/[A-Za-z]/)) {
        // Windows path with drive letter
        const drive = normalizedPath[0].toLowerCase();
        const rest = normalizedPath.substring(2);
        return {
            href: `file:///${drive}:${rest}`
        };
    } else if (normalizedPath.startsWith('//')) {
        // UNC path
        return {
            href: `file:${normalizedPath}`
        };
    } else {
        // Regular path
        return {
            href: `file://${normalizedPath}`
        };
    }
}

compileTarget();