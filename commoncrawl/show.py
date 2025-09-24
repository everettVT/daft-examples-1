# /// script
# dependencies = ["daft"]
# ///

import daft

if __name__ == "__main__":
    df = daft.read_warc("s3://commoncrawl/crawl-data/CC-MAIN-2025-33/segments/1754151579063.98/warc/CC-MAIN-20250815204238-20250815234238-00999.warc.gz")
    df.show()

