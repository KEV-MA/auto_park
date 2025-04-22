import os
import subprocess
import argparse
import shutil
from datetime import datetime

def convert_markdown_to_pdf(input_file, output_file, include_images=True):
    """
    Convert markdown to PDF using pandoc.
    Requires pandoc and a PDF engine (like wkhtmltopdf) to be installed.
    """
    print(f"Converting {input_file} to {output_file}...")
    
    # Create a temporary directory for processing
    temp_dir = "report/temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy markdown file to temp directory
    temp_md = os.path.join(temp_dir, "report.md")
    shutil.copy(input_file, temp_md)
    
    # If needed, copy images to temp directory
    if include_images:
        # Get the directory of the input file
        input_dir = os.path.dirname(input_file)
        
        # Copy all PNG files from the input directory to the temp directory
        for file in os.listdir(input_dir):
            if file.endswith(".png"):
                shutil.copy(os.path.join(input_dir, file), temp_dir)
        
        # Update image references in the markdown file
        with open(temp_md, 'r') as f:
            content = f.read()
        
        # Replace image references like ![Title](image.png) with ![Title](temp_dir/image.png)
        import re
        for match in re.finditer(r'!\[(.*?)\]\((.*?\.png)\)', content):
            img_title = match.group(1)
            img_path = match.group(2)
            img_filename = os.path.basename(img_path)
            
            # Only replace if the image path doesn't already include the temp directory
            if temp_dir not in img_path:
                content = content.replace(
                    f'![{img_title}]({img_path})',
                    f'![{img_title}]({img_filename})'
                )
        
        with open(temp_md, 'w') as f:
            f.write(content)
    
    # Run pandoc to convert markdown to PDF
    try:
        cmd = [
            "pandoc",
            temp_md,
            "--pdf-engine=wkhtmltopdf",
            "-o", output_file,
            "--standalone",
            "--toc",
            "--toc-depth=2",
            "--highlight-style=tango",
            "-V", "geometry:margin=1in",
            "-V", "title=Auto Park Evaluation Results",
            "-V", f"date={datetime.now().strftime('%B %d, %Y')}"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"PDF generated successfully: {output_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting markdown to PDF: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        
        # Alternative approach using markdown_to_pdf library if available
        try:
            print("Trying alternative conversion method...")
            from markdown_to_pdf import convert
            convert(temp_md, output_file)
            print(f"PDF generated successfully using alternative method: {output_file}")
        except ImportError:
            print("markdown_to_pdf library not available.")
            print("Please install pandoc and wkhtmltopdf for PDF conversion.")
            print("For Ubuntu/Debian: sudo apt-get install pandoc wkhtmltopdf")
            print("For macOS with Homebrew: brew install pandoc wkhtmltopdf")
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description='Generate PDF report from markdown results')
    parser.add_argument('--input', default='report/results.md', help='Input markdown file')
    parser.add_argument('--output', default='report/auto_park_evaluation.pdf', help='Output PDF file')
    args = parser.parse_args()
    
    convert_markdown_to_pdf(args.input, args.output)
    
    print("\nYou can also generate the PDF manually using:")
    print(f"pandoc {args.input} --pdf-engine=wkhtmltopdf -o {args.output} --standalone --toc")

if __name__ == "__main__":
    main() 