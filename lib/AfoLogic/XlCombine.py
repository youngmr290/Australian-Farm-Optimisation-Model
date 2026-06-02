'''
17/4/2021 by Michael Young
Module to combine selected excel documents into one.
Useful when generating Report.xl in multiple cloud instances.

Limitations:
    1. sheets with multiple tables are not fully supported.
    2. All excel files being combined need to be the same.
'''

import os
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __package__:
    from . import ReportFunctions as rep
else:
    sys.path.append(str(PROJECT_ROOT))
    from lib.AfoLogic import ReportFunctions as rep


output_path = PROJECT_ROOT / 'Output'
combined_filename = 'combined_file.xlsx'

# Files to combine this time
include_text = [
    'Report0_1400',
    'Report0_1200 v4b',
]


def combine_excel_files(
        output_path=output_path,
        combined_filename=combined_filename,
        include_text=include_text):
    """Combine selected report workbooks into a single workbook."""
    # Get selected report files
    selected_files = []

    for file in sorted(os.listdir(output_path)):
        if file.startswith('~') or not file.endswith('.xlsx') or file == combined_filename:
            continue

        if not any(text in file for text in include_text):
            continue

        selected_files.append(file)

    if not selected_files:
        raise ValueError('No Excel files selected to combine.')

    print('Combining the following files:')
    for file in selected_files:
        print(f'  {file}')

    excel_files = [
        pd.ExcelFile(output_path / file)
        for file in selected_files
    ]

    if 'df_settings' not in excel_files[0].sheet_names:
        raise ValueError(
            f"'{selected_files[0]}' does not contain a 'df_settings' sheet. "
            "Regenerate the report with the current report writer, or combine "
            "workbooks that include this metadata sheet."
        )

    missing_settings = [
        file
        for file, excel_file in zip(selected_files, excel_files)
        if 'df_settings' not in excel_file.sheet_names
    ]
    if missing_settings:
        raise ValueError(
            "The following selected workbooks do not contain a 'df_settings' sheet: "
            f"{', '.join(missing_settings)}"
        )

    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(
        output_path / combined_filename,
        engine='xlsxwriter'
    )


    # Loop through sheets and combine across selected excel files
    sheets = excel_files[0].sheet_names
    df_settings = excel_files[0].parse(
        sheet_name='df_settings',
        header=[0],
        index_col=[0]
    )

    for sheet in sheets:
        if sheet != 'df_settings':
            df_list = []

            l_index = list(range(df_settings.loc[sheet, 'index']))
            l_header = list(range(df_settings.loc[sheet, 'cols']))

            for report in excel_files:
                df = report.parse(
                    sheet_name=sheet,
                    header=l_header,
                    index_col=l_index
                )
                df_list.append(df)

            df_total = pd.concat(df_list)

            # Write to new excel file using custom function in ReportFunctions.py.
            rep.f_df2xl(writer, df_total, sheet, option=1)

    writer.close()

    print(f'Combined file written to {output_path / combined_filename}')


if __name__ == '__main__':
    combine_excel_files()
