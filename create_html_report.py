import os
import re
import argparse
from datetime import datetime

def convert_markdown_to_html(input_file, output_file):
    """
    Convert markdown to HTML with embedded images.
    """
    print(f"Converting {input_file} to {output_file}...")
    
    # Read markdown content
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Get the directory of the input file for image references
    input_dir = os.path.dirname(input_file)
    
    # Convert markdown headings
    content = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
    content = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
    content = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
    
    # Convert markdown lists
    content = re.sub(r'^- (.*?)$', r'<li>\1</li>', content, flags=re.MULTILINE)
    content = re.sub(r'(<li>.*?</li>\n)+', r'<ul>\n\g<0></ul>', content, flags=re.MULTILINE)
    
    # Convert bold text
    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    
    # Convert italic text
    content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
    
    # Handle paragraphs (any line that's not a heading, list, or blank)
    content = re.sub(r'^(?!<h|<ul|<li|$)(.*?)$', r'<p>\1</p>', content, flags=re.MULTILINE)
    
    # Handle image references
    def replace_image(match):
        img_text = match.group(1)
        img_path = match.group(2)
        
        # Get the full image path
        if not os.path.isabs(img_path):
            img_path = os.path.join(input_dir, img_path)
        
        # Check if image exists
        if os.path.exists(img_path):
            return f'<div class="image-container"><img src="{img_path}" alt="{img_text}" /><p class="caption">{img_text}</p></div>'
        else:
            return f'<p>[Image not found: {img_path}]</p>'
    
    content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image, content)
    
    # Create HTML document
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Auto Park Evaluation Results</title>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2980b9;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #3498db;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .caption {{
            font-style: italic;
            color: #666;
            margin-top: 5px;
        }}
        ul {{
            background-color: #f9f9f9;
            padding: 10px 20px;
            border-radius: 5px;
        }}
        li {{
            margin-bottom: 5px;
        }}
        strong {{
            color: #2c3e50;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            font-size: 0.8em;
            color: #777;
            border-top: 1px solid #ddd;
            padding-top: 10px;
        }}
    </style>
</head>
<body>
    {content}
    <div class="footer">
        <p>Generated on {datetime.now().strftime('%B %d, %Y')} | Auto Park Evaluation</p>
    </div>
</body>
</html>
'''
    
    # Write HTML to output file
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"HTML report generated successfully: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate HTML report from markdown results')
    parser.add_argument('--input', default='report/results.md', help='Input markdown file')
    parser.add_argument('--output', default='report/auto_park_evaluation.html', help='Output HTML file')
    args = parser.parse_args()
    
    convert_markdown_to_html(args.input, args.output)
    
    print(f"\nReport generated at: {args.output}")
    print(f"Open it in your browser to view the results.")

if __name__ == "__main__":
    main() 