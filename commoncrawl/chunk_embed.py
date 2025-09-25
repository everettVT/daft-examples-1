# /// script
# dependencies = ["daft", "torch", "sentence-transformers", "selectolax"]
# ///

import daft
from daft import col, DataType
from daft.functions import embed_text

from selectolax.parser import HTMLParser

@daft.func()
def remove_http_headers(x: str) -> str:
    """Remove HTTP headers from input string by splitting on double CRLF, returning the body or empty string."""
    if x is None:
        return ""
    if len(x.split("\r\n\r\n")) > 1:
        return x.split("\r\n\r\n")[1]
    return ""

@daft.func()
def extract_blocks(html: str) -> list[{"txt": str, "tag": str}]: 
    """Parse HTML using Selectolax, remove scripts/styles/noscripts, extract text blocks from key tags."""

    tree = HTMLParser(html)
    for n in tree.css("script,style,noscript"):
        n.decompose()

    all_nodes = tree.css("""title, article, main, p, h1, h2, h3, h4, h5, h6, li, div, section, img[alt], figcaption, caption, blockquote, table th, table td, pre, code, summary, meta[name="description"], meta[property="og:title"], meta[property="og:description"]""")
    blocks = [
        {"txt": node.text(separator=" ", strip=True), "tag": node.tag} 
        for node in all_nodes if node.text(separator=" ", strip=True)
    ]
    return blocks

if __name__ == "__main__":
    URI = "s3://commoncrawl/crawl-data/CC-MAIN-2025-33/segments/1754151579063.98/warc/CC-MAIN-20250815204238-20250815234238-00999.warc.gz"
    PAGE_LIMIT = 10

    df = (
        daft.read_warc(URI).limit(PAGE_LIMIT)
        .where(col("WARC-Identified-Payload-Type")== "text/html")
        .with_column("content_raw", remove_http_headers(col("warc_content").try_decode("utf-8")))
        .where(col("content_raw") != "")
        .with_column("blocks", extract_blocks(col("content_raw")))
        .explode("blocks")
        .where(col("blocks")["txt"] != "")
        .where(col("blocks")["txt"].not_null())
        .with_column("block_id", col("WARC-Record-ID")+ "-" + col("blocks")["tag"] + "-" + col("blocks")["txt"].hash()) # Record ID + Tag + Text Hash
        .with_column("text", col("blocks")["txt"])
    )
 
    #df = df.with_column("embedding", embed_text(col("warc_content")))
    df.show()

