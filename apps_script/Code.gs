/**
 * BELLA Center Experiment Log — Google Apps Script
 *
 * This script is deployed as a Google Apps Script project and called from
 * Python via the Apps Script API (see logmaker_4_googledocs/docgen.py).
 * The deployed script is identified by the SCRIPT_ID stored in
 * logmaker_4_googledocs/config.ini.
 *
 * To update the deployed script:
 *   1. Open https://script.google.com and select the project.
 *   2. Paste the updated contents of this file into the editor.
 *   3. Deploy a new version (Deploy → Manage deployments → New version).
 *   4. The SCRIPT_ID in config.ini does not change between versions.
 *
 * Heading format used throughout (Heading 3):
 *   YY-MM-DD (HH:MM:SS) Scan NNN[: optional note]
 *
 * Originally by Tobias Ostermayr (2020). Extended at BELLA Center, LBNL.
 */


// ─── Log file management ────────────────────────────────────────────────────

/**
 * Create (or find) a daily experiment log document from a template.
 * Returns the Google Doc ID of the new or existing document.
 */
function createExperimentLog(templateID, templatefolderID, documentfolderID, filename) {
  var file = DriveApp.getFileById(templateID);
  var dest_folder = DriveApp.getFolderById(documentfolderID);
  var documentIDtest = checkFile(documentfolderID, filename);
  if (documentIDtest != 'nothing here') {
    return documentIDtest;
  }
  var copiedDocument = file.makeCopy(filename, dest_folder);
  return copiedDocument.getId();
}


// ─── Document editing ───────────────────────────────────────────────────────

/**
 * Append all elements from a template document to a target document.
 * Used to add new scan-entry blocks to an existing log.
 */
function appendTemplate(templateID, documentID) {
  var templateBody = DocumentApp.openById(templateID).getBody();
  var thisBody = DocumentApp.openById(documentID).getBody();
  for (var i = 0; i < templateBody.getNumChildren(); i++) {
    switch (templateBody.getChild(i).getType()) {
      case DocumentApp.ElementType.PARAGRAPH:
        thisBody.appendParagraph(templateBody.getChild(i).copy());
        break;
      case DocumentApp.ElementType.LIST_ITEM:
        thisBody.appendListItem(templateBody.getChild(i).copy());
        break;
      case DocumentApp.ElementType.TABLE:
        thisBody.appendTable(templateBody.getChild(i).copy());
        break;
      case DocumentApp.ElementType.INLINE_IMAGE:
        thisBody.appendImage(templateBody.getChild(i).copy());
        break;
    }
  }
  return 'success';
}

/**
 * Replace {{placeholder}} tokens in a document with provided values.
 */
function findAndReplace(documentID, keys, values) {
  var thisBody = DocumentApp.openById(documentID).getBody();
  for (var i = 0; i < keys.length; i++) {
    thisBody.replaceText('(?i)' + '{{' + keys[i] + '}}', values[i]);
  }
  return '(?i)' + '{{' + keys[1] + '}}';
}

/**
 * Replace a text placeholder inside a table with a Drive image.
 * Only searches inside tables.
 */
function findAndReplaceImage(documentID, imageID, placeholder) {
  var thisBody = DocumentApp.openById(documentID).getBody();
  var tables = thisBody.getTables();
  for (var k in tables) {
    var table = tables[k];
    for (var row = 0; row < table.getNumRows(); row++) {
      var tablerow = table.getRow(row);
      for (var cell = 0; cell < tablerow.getNumCells(); cell++) {
        if (tablerow.getChild(cell).getText() == placeholder) {
          var blob = DriveApp.getFileById(imageID).getBlob();
          table.replaceText('(?i)' + placeholder, '');
          table.getCell(row, cell).insertImage(0, blob);
        }
      }
    }
  }
  return 0;
}

/**
 * Insert a Drive image into a specific cell of the 2×2 display table
 * (tables[1]) under the Heading 3 for the given scan number.
 *
 * @param {string} documentID  Google Doc ID.
 * @param {number} scanNumber  Scan number used to locate the section heading.
 * @param {number} row         Zero-based row index in the display table.
 * @param {number} column      Zero-based column index in the display table.
 * @param {string} imageID     Drive file ID of the image to insert.
 * @returns {string} 'success' or an error description.
 */
function insertImageToTableCell(documentID, scanNumber, row, column, imageID) {
  var body = DocumentApp.openById(documentID).getBody();
  var headingIndex = _findScanHeadingIndex(body, scanNumber);
  if (headingIndex < 0) {
    Logger.log('Heading for Scan ' + scanNumber + ' not found.');
    return 'Heading for Scan ' + scanNumber + ' not found.';
  }

  var tables = _tablesInSection(body, headingIndex);
  if (tables.length < 2) {
    Logger.log('No display table (tables[1]) found for Scan ' + scanNumber);
    return 'No display table found for Scan ' + scanNumber;
  }

  var table = tables[1];
  if (row >= table.getNumRows() || column >= table.getRow(row).getNumCells()) {
    Logger.log('Invalid row/column for Scan ' + scanNumber);
    return 'Invalid row or column index.';
  }

  var cell = table.getRow(row).getCell(column);
  cell.clear();
  cell.insertImage(0, DriveApp.getFileById(imageID).getBlob());
  Logger.log('Image inserted: Scan ' + scanNumber + ' [' + row + ',' + column + ']');
  return 'success';
}

/**
 * Append a hyperlink paragraph at the end of the scan section for scanNumber.
 *
 * The paragraph is inserted just before the next Heading 3 (or at the end of
 * the document if this is the last scan entry).  Multiple calls are safe and
 * purely additive — each call appends one new paragraph.
 *
 * @param {string} documentID  Google Doc ID.
 * @param {number} scanNumber  Scan number used to locate the section heading.
 * @param {string} label       Visible link text (e.g. "UC_GaiaMode: summary.png").
 * @param {string} url         URL the link points to (e.g. a Drive view URL).
 * @returns {string} 'success' or an error description.
 */
function appendLinkToScan(documentID, scanNumber, label, url) {
  var body = DocumentApp.openById(documentID).getBody();
  var headingIndex = _findScanHeadingIndex(body, scanNumber);
  if (headingIndex < 0) {
    Logger.log('Heading for Scan ' + scanNumber + ' not found.');
    return 'Heading for Scan ' + scanNumber + ' not found.';
  }

  var insertIndex = _sectionEndIndex(body, headingIndex);

  var newPara = body.insertParagraph(insertIndex, label);
  newPara.editAsText().setLinkUrl(0, label.length - 1, url);
  Logger.log('Appended link for Scan ' + scanNumber + ': ' + label);
  return 'success';
}


// ─── Spreadsheet helpers ────────────────────────────────────────────────────

/**
 * Return the last non-empty row (as display values) from a sheet range.
 */
function lastRowFromSpreadsheet(fileID, sheetString, firstCol, lastCol) {
  var sheet = SpreadsheetApp.openById(fileID).getSheetByName(sheetString);
  var rng = sheet.getRange(firstCol + ':' + lastCol).getValues();
  var lrIndex = 0;
  for (var i = rng.length - 1; i >= 0; i--) {
    lrIndex = i;
    if (!rng[i].every(function (c) { return c == ''; })) break;
  }
  return sheet.getRange(firstCol + (lrIndex + 1) + ':' + lastCol + (lrIndex + 1)).getDisplayValues();
}


// ─── Helper functions ───────────────────────────────────────────────────────

/**
 * Check whether a file with the given name exists in a Drive folder.
 * Returns the file ID if found, or 'nothing here'.
 */
function checkFile(folderID, filename) {
  var files = DriveApp.getFolderById(folderID).searchFiles('title contains "' + filename + '"');
  if (files.hasNext()) {
    return files.next().getId();
  }
  return 'nothing here';
}

/**
 * Return true if the document body contains the search string, else false.
 */
function checkFileContains(fileID, search) {
  return DocumentApp.openById(fileID).getBody().findText(search) !== null;
}

/**
 * Find the body child index of the Heading 3 for the given scan number.
 * Returns -1 if not found.
 */
function _findScanHeadingIndex(body, scanNumber) {
  var regex = new RegExp(
    '^\\d{2}-\\d{2}-\\d{2} \\(\\d{2}:\\d{2}:\\d{2}\\) Scan ' + scanNumber + '(:.*)?$'
  );
  var n = body.getNumChildren();
  for (var i = 0; i < n; i++) {
    var child = body.getChild(i);
    if (child.getType() !== DocumentApp.ElementType.PARAGRAPH) continue;
    var para = child.asParagraph();
    if (para.getHeading() === DocumentApp.ParagraphHeading.HEADING3 && regex.test(para.getText())) {
      return i;
    }
  }
  return -1;
}

/**
 * Collect all Table elements between headingIndex (exclusive) and the next
 * Heading 3 (exclusive), in document order.
 */
function _tablesInSection(body, headingIndex) {
  var tables = [];
  var n = body.getNumChildren();
  for (var j = headingIndex + 1; j < n; j++) {
    var child = body.getChild(j);
    if (child.getType() === DocumentApp.ElementType.TABLE) {
      tables.push(child.asTable());
    } else if (child.getType() === DocumentApp.ElementType.PARAGRAPH) {
      if (child.asParagraph().getHeading() === DocumentApp.ParagraphHeading.HEADING3) break;
    }
  }
  return tables;
}

/**
 * Return the child index at which new content should be inserted to appear
 * at the END of the scan section starting at headingIndex — i.e. just before
 * the next Heading 3, or at the end of the document if none follows.
 */
function _sectionEndIndex(body, headingIndex) {
  var n = body.getNumChildren();
  for (var j = headingIndex + 1; j < n; j++) {
    var child = body.getChild(j);
    if (child.getType() === DocumentApp.ElementType.PARAGRAPH &&
        child.asParagraph().getHeading() === DocumentApp.ParagraphHeading.HEADING3) {
      return j;
    }
  }
  return n;
}
