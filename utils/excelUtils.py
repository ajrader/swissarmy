import xlrd
import pyxlsb
import pandas as pd


def get_excel_sheet_tab_names(filename, **kwargs):
    workbook = xlrd.open_workbook(filename,**kwargs)
    worksheet_tab_names = workbook.sheet_names()
    return worksheet_tab_names


def xlsb_to_df(filename, sheet_name = None, skip_row=False, header=True):
    if skip_row:
        n_row_skip = skip_row
    else:
        n_row_skip = 0

    #if header:
    #    header_row = n_row_skip
    #else:
    #    header_row = 0

    krows = 0
    my_data = []
    with pyxlsb.open_workbook(filename) as wb:
        if sheet_name is not None:
            sheet = wb.get_sheet(sheet_name)
        else:
            sheet = wb.get_sheet(1) # 1-based to match VBA using 1st index
        n_cols = sheet.dimension.w
        n_rows = sheet.dimension.h

        for row in sheet.rows():
            if krows < n_row_skip:
                pass
            else:
                row_values = [data.v for data in row]
                my_data.append(row_values)

        sheet.close()

    if header:
        labels = my_data.pop(0)
    else:
        labels = ['col_'+str(i) for i in n_cols]

    df = pd.DataFrame(data =my_data, columns = labels)

    return df


