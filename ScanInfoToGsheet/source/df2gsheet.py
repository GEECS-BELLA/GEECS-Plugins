import pygsheets

def df2gsheet(sheet_title, df, gdrive_dir):
    '''
    Open (Create if not exist) a google sheet in the google drive, then write down a given dataframe
    '''
    gc = pygsheets.authorize(service_file = 'service_account.json')

    #if can't find an exisiting google sheet, create it
    try:
        sheet = gc.open(sheet_title)
    except pygsheets.SpreadsheetNotFound as error:
        sheet = gc.create(sheet_title, folder = gdrive_dir)

    sheet[0].set_dataframe(df, (1,1))
    return sheet