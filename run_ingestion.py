# run_ingestion.py
from pathlib import Path
from src.ingest import ingest_book, RAW_DIR

print("Starting data ingestion...")
for book in RAW_DIR.glob("*.xls*"):
    print(f"Processing Excel file: {book.name}")
    try:
        ingest_book(book)
    except Exception as e:
        print(f"Error processing {book.name}: {e}")
print("Data ingestion complete.")