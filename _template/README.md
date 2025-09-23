# Daft Example

## Setup

To Clone this example, run the following command:

```bash
git clone https://github.com/Eventual-Inc/daft-examples/_template.git
```

Then to activate a virtual environment and install dependencies, just run:

```bash
uv sync
```

This example explores how to download pdfs from a remote filesystem. 

```python
import daft 
from daft import col


# Helper Functions and UDFs
@daft.func()
def foo(str: str) -> str:
    return str + " bar"

# Config
uri = "hf://datasets/Eventual-Inc/sample-files/papers/*.pdf"
row_limit = 10

# Dataframe Operations
df = (
    daft.from_glob_path(uri) 
    .with_column("foobar_path", foo(col("path"))) # Apply a function to the path column
    .with_column("document", col("path").url.download()) # Download documents
    .limit(row_limit)
)

# Display the first 8 results
df.show()
```
