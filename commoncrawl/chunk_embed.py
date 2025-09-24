# /// script
# dependencies = ["daft", "torch", "sentence-transformers"]
# ///

import daft
from daft.functions import embed_text

if __name__ == "__main__":
    df = daft.read_warc("s3://commoncrawl/crawl-data/CC-MAIN-2025-33/segments/1754151579063.98/warc/CC-MAIN-20250815204238-20250815234238-00999.warc.gz")
    df = df.where(daft.col("WARC-Type") == "response")
    df = df.with_column("embedding", embed_text(daft.col("warc_content")))
    df.show()

