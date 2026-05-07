import PyPDF2
import pandas as pd


class RAGPDFExtractor:
    """
    Extracts all potentially RAG-relevant content from a PDF.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_text = ""
        self.paragraphs = []

    def read_pdf(self):
        """
        Reads all text from the PDF.
        """
        with open(self.file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n=== Page {page_num} ===\n" + page_text
            self.raw_text = text
        return self.raw_text

    def extract_paragraphs(self):
        """
        Splits raw text into paragraphs or blocks of text that could be RAG-relevant.
        """
        if not self.raw_text:
            self.read_pdf()

        # Split by double line breaks or section headers
        raw_blocks = self.raw_text.split("\n\n")
        self.paragraphs = [block.strip() for block in raw_blocks if block.strip()]
        return self.paragraphs

    def to_dataframe(self):
        """
        Converts extracted text blocks into a pandas DataFrame for RAG analysis.
        """
        if not self.paragraphs:
            self.extract_paragraphs()

        df = pd.DataFrame(
            {
                "block_number": list(range(1, len(self.paragraphs) + 1)),
                "text": self.paragraphs,
                "RAG_status": [""]
                * len(self.paragraphs),  # Placeholder for RAG tagging later
            }
        )
        return df


# Example CLI usage
def main():
    pdf_file = "raw/rules/laws_games.pdf"
    extractor = RAGPDFExtractor(pdf_file)

    print("Reading PDF...")
    extractor.read_pdf()

    print("Extracting paragraphs/blocks...")
    blocks = extractor.extract_paragraphs()
    print(f"Extracted {len(blocks)} text blocks.")

    df = extractor.to_dataframe()
    print(df.head())

    output_csv = "rag_output.csv"
    df.to_csv(output_csv, index=False)
    print(f"Extracted blocks saved to {output_csv}")


if __name__ == "__main__":
    main()
