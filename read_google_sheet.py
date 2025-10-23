import gspread
from google.oauth2.service_account import Credentials
import random

# Define the scope
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

def read_prompt_column(sheet_url_or_id, worksheet_name=None):
    """
    Read the 'prompt' column from a Google Sheet and return as a list.

    Args:
        sheet_url_or_id: Google Sheet URL or ID
        worksheet_name: Name of the worksheet (if None, uses first sheet)

    Returns:
        List of values from the 'prompt' column
    """
    # Authenticate
    path_to_credential = '/Users/wangez/Downloads/GCP_Credentials/agolis-allen-first-13f3be86c3d1.json'
    creds = Credentials.from_service_account_file(path_to_credential, scopes=SCOPES)
    client = gspread.authorize(creds)


    # import google.auth
    # credentials, _ = google.auth.default()
    # client = gspread.authorize(credentials)

    # Open the spreadsheet
    if sheet_url_or_id.startswith('http'):
        sheet = client.open_by_url(sheet_url_or_id)
    else:
        sheet = client.open_by_key(sheet_url_or_id)

    # Get the worksheet
    if worksheet_name:
        worksheet = sheet.worksheet(worksheet_name)
    else:
        worksheet = sheet.get_worksheet(0)  # First sheet

    # Get all records as list of dictionaries
    records = worksheet.get_all_records()

    # Extract the 'prompt' column
    prompt_list = [record.get('Prompt', '') for record in records]

    return prompt_list


def add_random_column(sheet_url_or_id, random_numbers, worksheet_name=None, column_name='fc'):
    """
    Add random numbers to a new column in the sheet.

    Args:
        sheet_url_or_id: Google Sheet URL or ID
        random_numbers: List of random numbers to add
        worksheet_name: Name of the worksheet (if None, uses first sheet)
        column_name: Name of the new column to create (default: 'fc')
    """
    # Authenticate
    path_to_credential = '/Users/wangez/Downloads/GCP_Credentials/agolis-allen-first-13f3be86c3d1.json'
    creds = Credentials.from_service_account_file(path_to_credential, scopes=SCOPES)
    client = gspread.authorize(creds)

    # Open the spreadsheet
    if sheet_url_or_id.startswith('http'):
        sheet = client.open_by_url(sheet_url_or_id)
    else:
        sheet = client.open_by_key(sheet_url_or_id)

    # Get the worksheet
    if worksheet_name:
        worksheet = sheet.worksheet(worksheet_name)
    else:
        worksheet = sheet.get_worksheet(0)  # First sheet

    # Find the next available column
    header_row = worksheet.row_values(1)
    next_col = len(header_row) + 1

    # Add column header
    worksheet.update_cell(1, next_col, column_name)

    # Format random numbers as list of lists for batch update
    random_numbers_formatted = [[num] for num in random_numbers]

    # Update cells starting from row 2 (after header)
    num_rows = len(random_numbers)
    cell_range = f'{gspread.utils.rowcol_to_a1(2, next_col)}:{gspread.utils.rowcol_to_a1(num_rows + 1, next_col)}'
    worksheet.update(cell_range, random_numbers_formatted)

    print(f"Added {num_rows} random numbers to column '{column_name}'")


if __name__ == '__main__':
    # Example usage
    SHEET_URL = 'https://docs.google.com/spreadsheets/d/1pw_TnNL3OpYUaG_NDhsJJrlGZ1tZTbmQjBb8ef7AVm0/edit?resourcekey=0-q6-OmNTETqLUIAg0g15qqA&gid=0#gid=0'

    try:
        print("Starting to read Google Sheet...")
        prompts = read_prompt_column(SHEET_URL)
        print(f"Found {len(prompts)} prompts")

        print("\nGenerating random numbers for all prompts...")
        random_numbers = []
        for i, prompt in enumerate(prompts, 1):
            # Generate random number for each prompt
            random_num = random.random()
            random_numbers.append(random_num)
            print(f"Processed {i}/{len(prompts)}: Generated {random_num:.6f}")

        print(f"\nCollected {len(random_numbers)} random numbers")
        print("Updating sheet with all numbers in one batch...")
        add_random_column(SHEET_URL, random_numbers)
        print("Done!")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
