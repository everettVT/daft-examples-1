# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft", "pymupdf", "sentence_transformers", "spacy"]
# ///

import daft
from daft import col, DataType
from daft.functions import embed_text, unnest
import io
import pymupdf

@daft.func(return_dtype=
    DataType.list(DataType.struct({
        "page_number": DataType.int32(), 
        "text": DataType.string()
    }))
)
def extract_pdf_as_html(doc: bytes):
    """
    Parse PDF bytes into string text using PyPDF2.
    
    Args:
        doc: PDF document bytes
        
    Returns:
        Extracted text as string
    """
    content = []
  
    try:
        # Create a PDF reader object using BytesIO
        document = pymupdf.Document(stream=io.BytesIO(doc), filetype="pdf")
        
        # Extract text from all pages, track page number
        for page_number, page in enumerate(document):
            text = page.get_text("text") 
            content.append({
                "page_number": page_number,
                "text": text
            })

    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return content
    
    return content

@daft.udf(return_dtype=DataType.list(DataType.struct({
    "sent_id": DataType.int32(), 
    "sent_text": DataType.string(), 
    "sent_start": DataType.int32(), 
    "sent_end": DataType.int32(), 
    "sent_ents": DataType.list(DataType.string())
})))
class SpaCyChunkText:
    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_trf")

    def __call__(self, texts):
        sentence_texts = []
        for text in texts:
            doc = self.nlp(text)
            sentence_texts.append([{
                    "sent_id": i,
                    "sent_start": sent.start,
                    "sent_end": sent.end,
                    "sent_text": sent.text,
                    "sent_ents": [ent.text for ent in sent.ents] if sent.ents else [],
                }  for i, sent in enumerate(doc.sents)]
            )
            
        return sentence_texts

        
if __name__ == "__main__":

    MODEL_ID = "google/embeddinggemma-300m"
    MAX_DOCS = 1

    # Config
    uri = "hf://datasets/Eventual-Inc/sample-files/papers/*.pdf"

    # Discover and download pdfs
    df = (
        daft.from_glob_path(uri).limit(MAX_DOCS)
        .with_column("documents", col("path").url.download()) 
    )
    #df.show() # Optionally show the dataframe

    # Extract text from pdf pages
    df = (
        df
        .with_column("pages", extract_pdf_as_html(col("documents")))
        .explode("pages")
        .select(col("path"), unnest(col("pages")))
    )
    #df.show()

    # Chunk page text into sentences
    df = (
        df
        .with_column("text_normalized", col("text").normalize(nfd_unicode=True, white_space=True))
        .with_column("sentences", SpaCyChunkText(col("text_normalized")))
        .explode("sentences")
        .select(col("path"), col("page_number"), unnest(col("sentences")))
        .where(col("sent_end") - col("sent_start") > 1) # remove sentences that are too short
    )   
    #df.show()

    # Embed sentences
    df = (
        df
        .with_column(f"text_embed_{MODEL_ID.split('/')[1]}", embed_text(col("sent_text"), provider="sentence_transformers", model=MODEL_ID))
    )
    #df.show()

    # Write to parquet
    df.write_parquet(f"data/eventual_sample_pdf_sentence_embeddings.parquet")

    df.show()


