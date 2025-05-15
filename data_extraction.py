import pdfplumber
import pandas as pd
import json
import re
import sys

# Function to clean and structure extracted text
def clean_and_structure_text(text):
    structured_data = {
        "sections": []
    }

    lines = text.split("\n")

    # Use regex to identify section headers and their content
    current_section = None
    section_pattern = re.compile(r"^(\d+\.\s.*)")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        section_match = section_pattern.match(line)
        if section_match:
            # Start a new section
            current_section = {
                "header": section_match.group(1),
                "content": []
            }
            structured_data["sections"].append(current_section)
        elif current_section:
            # Add line to the current section content
            current_section["content"].append(line)

    # Combine content lines into paragraphs
    for section in structured_data["sections"]:
        section["content"] = " ".join(section["content"])

    return structured_data

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    extracted_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text += page.extract_text() + "\n"

    return extracted_text

# Main function to process PDF and save structured data
def process_pdf_to_json(pdf_path, output_json_path):
    raw_text = extract_text_from_pdf(pdf_path)
    structured_data = clean_and_structure_text(raw_text)

    # Save structured data to JSON
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(structured_data, json_file, indent=4, ensure_ascii=False)

    print(f"Structured data saved to {output_json_path}")


def clean_new_lines_table(table):
    for row in range(len(table)):
        for column in range(len(table[row])):
            if table[row][column] is not None:
                table[row][column] = table[row][column].replace("\n", " ")

def count_non_empty_cells_per_row(row):
    count = 0
    first_non_empty_idx = -1
    for idx, cell in enumerate(row):
        if cell is not None and cell != '':
            count += 1
            if first_non_empty_idx == -1:
                first_non_empty_idx = idx

    return first_non_empty_idx, count

def extract_cpdlc_messages_table(table, category, page_num):
    clean_new_lines_table(table)

    ref_idx = table[0].index("Ref #") if "Ref #" in table[0] else -1
    message_int_idx = table[0].index("Message Intent/Use") if "Message Intent/Use" in table[0] else -1
    message_ele_idx = table[0].index("Message Element") if "Message Element" in table[0] else -1
    response_idx = table[0].index("Resp.") if "Resp." in table[0] else -1

    if ref_idx == -1 or message_int_idx == -1 or message_ele_idx == -1 or response_idx == -1:
        print(f"Page: {page_num} - Table: {table[0]}")
        return None, category

    category = category
    messages = []
    for row in table[1:]:
        first_non_empty_cell, non_empty_cells = count_non_empty_cells_per_row(row)
        if first_non_empty_cell == -1:
            continue

        if non_empty_cells > 0 and non_empty_cells < 4:
            if first_non_empty_cell < 4:
                category = row[first_non_empty_cell]
            continue

        messages.append([row[ref_idx], row[message_int_idx], row[message_ele_idx], row[response_idx], category])

    return messages, category


def extract_tables_from_pdf(pdf_path):
    tables_df = pd.DataFrame(columns=["Ref_Num", "Message_Intent", "Message_Element", "Response", "Category"])
    category = "Responses/Acknowledgements (uplink)"
    messages_rows = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages[3:]):

            tables = page.extract_tables()
            for table in tables:
                messages, category = extract_cpdlc_messages_table(table, category, page_num)
                if messages is not None:
                    messages_rows.extend(messages)

    return pd.DataFrame(messages_rows, columns=["Ref_Num", "Message_Intent", "Message_Element", "Response", "Category"]).to_dict(orient="records")


if __name__ == '__main__':
    args = sys.argv
    if(len(args) < 2):
        file_path = input("Please enter the path of the CPDLC pdf file containing the tables: ")
    else:
        file_path = sys.argv[1]
    tables = extract_tables_from_pdf(file_path)

    # Save extracted tables as JSON
    with open("tables.json", "w", encoding="utf-8") as json_file:
        json.dump(tables, json_file, indent=4, ensure_ascii=False)

    print(f"Extracted {len(tables)} tables and saved to tables.json")