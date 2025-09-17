#!/bin/bash

# Script to create a clean zip file of the project
# This excludes large files that can be regenerated

echo "🧹 Cleaning up project for zipping..."

# Create a temporary directory for the clean project
TEMP_DIR="price_algo_clean"
ZIP_FILE="price_algo.zip"

# Remove existing temp directory and zip file
rm -rf "$TEMP_DIR" "$ZIP_FILE"

# Create clean directory
mkdir "$TEMP_DIR"

echo "📁 Copying project files (excluding large files)..."

# Copy all files except those in .zipignore
rsync -av --exclude-from=.zipignore . "$TEMP_DIR/"

# Remove the .zipignore file from the clean copy
rm -f "$TEMP_DIR/.zipignore"

echo "📦 Creating zip file..."

# Create zip file
cd "$TEMP_DIR"
zip -r "../$ZIP_FILE" . -x "*.DS_Store" "*/__pycache__/*" "*/.*"
cd ..

# Clean up temp directory
rm -rf "$TEMP_DIR"

# Show zip file size
echo "✅ Zip file created: $ZIP_FILE"
echo "📊 File size: $(du -sh "$ZIP_FILE" | cut -f1)"

echo ""
echo "🎯 What was excluded:"
echo "   - .venv/ directory (985MB) - Python virtual environment"
echo "   - mlruns/ directory (4MB) - MLflow artifacts"
echo "   - mlflow.db (220KB) - MLflow database"
echo "   - __pycache__/ directories - Python cache files"
echo "   - .DS_Store files - macOS system files"
echo ""
echo "💡 To restore the project:"
echo "   1. Unzip the file"
echo "   2. Run: python -m venv .venv"
echo "   3. Run: source .venv/bin/activate (or .venv\\Scripts\\activate on Windows)"
echo "   4. Run: pip install -e '.[dev]'"
echo "   5. Run: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000"
echo "   6. Run: python src/training/train.py"
echo "   7. Run: python run_service.py"
