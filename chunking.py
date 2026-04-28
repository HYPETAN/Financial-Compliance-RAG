import os
import re
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def clean_universal_sec_html(file_path):
    """
    A universal parser for SEC EDGAR full-submission.txt files.
    """
    print(f"\n--- Loading Document ---")
    print(f"Target: {file_path}")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_content = f.read()

    print(f"Original size: {len(raw_content):,} characters.")

    # STEP 1: Macro-Isolation (Extract ONLY the 10-K from the 100+ documents in the bundle)
    # This looks for <DOCUMENT>, then <TYPE>10-K, and grabs everything until </DOCUMENT>
    doc_pattern = re.compile(
        r'<DOCUMENT>\s*<TYPE>10-K.*?</DOCUMENT>', re.IGNORECASE | re.DOTALL)
    match = doc_pattern.search(raw_content)

    if match:
        html_block = match.group(0)
        print(
            f"Successfully isolated the 10-K document (Size: {len(html_block):,} chars).")
    else:
        print("[WARNING] Could not find <TYPE>10-K. Proceeding with full file.")
        html_block = raw_content

    # STEP 2: Targeted Excision (Nuke the accounting dictionary, keep the inline text)
    # Destroy the <ix:header> block completely
    html_block = re.sub(r'<ix:header.*?</ix:header>', '',
                        html_block, flags=re.IGNORECASE | re.DOTALL)
    # Destroy explicitly hidden divs where SEC hides more taxonomy data
    html_block = re.sub(r'<div[^>]*style="[^"]*display:\s*none[^"]*"[^>]*>.*?</div>',
                        '', html_block, flags=re.IGNORECASE | re.DOTALL)

    # STEP 3: DOM Stripping (BeautifulSoup)
    soup = BeautifulSoup(html_block, "lxml")

    # Remove tables (they chunk terribly), scripts, and styles
    for element in soup(["script", "style", "table"]):
        element.decompose()

    # Extract the clean text.
    # Using "\n\n" as a separator ensures paragraphs stay separated.
    text = soup.get_text(separator="\n\n")

    # Clean up excessive empty lines
    clean_text = '\n'.join(line for line in text.splitlines() if line.strip())

    print(f"Final Cleaned text size: {len(clean_text):,} characters.")
    return clean_text


def chunk_clean_text(clean_text, source_meta):
    """
    Semantically chunks the cleaned text.
    """
    print("Chunking text semantically...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "(?<=\\. )", " ", ""],
        length_function=len
    )

    raw_chunks = text_splitter.split_text(clean_text)
    documents = [Document(page_content=chunk, metadata={
                          "source": source_meta}) for chunk in raw_chunks]

    print(f"[SUCCESS] Created {len(documents)} semantic chunks.\n")
    return documents


if __name__ == "__main__":
    # Point this to your downloaded full-submission.txt
    aapl_dir = os.path.join("sec_data", "sec-edgar-filings", "AAPL", "10-K")

    target_file = None
    for root, dirs, files in os.walk(aapl_dir):
        for f in files:
            if f.endswith('.txt') or f.endswith('.html'):
                target_file = os.path.join(root, f)
                break
        if target_file:
            break

    if target_file:
        # Execute the universal pipeline
        clean_text = clean_universal_sec_html(target_file)
        resulting_chunks = chunk_clean_text(
            clean_text, source_meta="AAPL_10-K_Latest")

        # Inspect the chunk to prove the gibberish is gone
        if len(resulting_chunks) > 10:
            print("--- INSPECTING CHUNK 10 ---")
            print(resulting_chunks[10].page_content)
    else:
        print("Could not find the target text file.")
