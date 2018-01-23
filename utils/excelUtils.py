import xlrd


def get_excel_sheet_tab_names(filename, **kwargs):
    workbook = xlrd.open_workbook(filename,**kwargs)
    worksheet_tab_names = workbook.sheet_names()
    return worksheet_tab_names


