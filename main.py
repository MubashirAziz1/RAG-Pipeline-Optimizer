
import asyncio
import sys
from pathlib import Path
from src.services.pdf_parser.factory import make_pdf_parser_service


async def parse_and_display(pdf_path: str):
    """Parse PDF and display the extracted content."""
    
    file_path = Path(pdf_path)
    
    # Validate file
    if not file_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return
    
    if not file_path.suffix.lower() == '.pdf':
        print(f"❌ Not a PDF file: {pdf_path}")
        return
    
    print(f"\n📄 Parsing: {file_path.name}\n")
    
    try:
        # Get parser and parse the file
        parser = make_pdf_parser_service()
        result = await parser.parse_pdf(file_path)
        
        if not result:
            print("❌ Parsing failed - no result returned")
            return
        
        # Print the extracted content
        print("=" * 80)
        print("EXTRACTED CONTENT")
        print("=" * 80)
        print(result.raw_text)
        print("=" * 80)
        
        # Print statistics
        print(f"\n📊 Statistics:")
        print(f"   - Total characters: {len(result.raw_text)}")
        print(f"   - Total sections: {len(result.sections)}")
        print(f"   - Total figures: {len(result.figures)}")
        print(f"   - Total tables: {len(result.tables)}")
        print(f"   - Parser used: {result.parser_used}")
        
        print("\n✅ Parsing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        print("Example: python main.py document.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    asyncio.run(parse_and_display(pdf_path))