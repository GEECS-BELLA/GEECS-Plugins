//HTT EXPERIMENT LOG GENERATOR

/**
 * Extending Google Docs developer guide:
 *     https://developers.google.com/apps-script/guides/docs
 *
 * Document service reference documentation:
 *     https://developers.google.com/apps-script/reference/document/
 *
 *
 * This project works with a local python interface calling the functions via Google Apps Scripts API.
 * The purpose of this script is to make automated experiment logs for the HTT experiment at the Berkeley Lab Laser Accelerator Center.
 *
 * by Tobias Ostermayr, last updated August 6, 2020
 */

//=================================MAIN FUNCTIONS======================================

//--------------------------------LOG FILE CREATOR-------------------------------------
//If no Logfile has been created that day, this function copies the template logfile into the Experiment Log folder and returns its ID.
//If the Logfile does exist already, this function returns its ID
// function createExperimentLog(templateID,templatefolderID,documentfolderID,filename) {
//   var date = Utilities.formatDate(new Date(), 'America/Los_Angeles', "MM-dd-YY");
//   var file = DriveApp.getFileById(templateID);
//   var source_folder = DriveApp.getFolderById(templatefolderID);
//   var dest_folder = DriveApp.getFolderById(documentfolderID);
//   var documentIDtest = checkFile(documentfolderID, filename);
//   //If the file exists in the drive return its ID. Else, create the file and return its ID.
//   if(documentIDtest != "nothing here"){
//   return documentIDtest
//   }else{
//   var file2 = file.makeCopy(filename);
//   dest_folder.addFile(file2);
//   source_folder.removeFile(file2);
//   var documentID = file2.getId();
//   return documentID
//  }
// }


// Sams attempt at making this bit of code work on shared drives. Currently, this does seem to work.
// the first function is for live testing in goolge scripts env, commented out for now
// function test_createExperimentLog() {
//   var docId = '1-GpzJW1seWBWhCOixXVb47SSYxEit2PEXWJceincEck';
//   var folderId = '16UrOep7RaFTrVUdbsx0tEJTEfK1GUW_C';
//   var templatefolderID = '1-9fFZlaFl66bEkOwmxkRjyAu4QSpXlrZ';
//   var filename = 'HTU test'

//   var result = createExperimentLog(docId, templatefolderID, folderId,  filename);
//   Logger.log(result);
// }

function createExperimentLog(templateID, templatefolderID, documentfolderID, filename) {
  var date = Utilities.formatDate(new Date(), 'America/Los_Angeles', "MM-dd-YY");

  // Access the template file and folders directly using their IDs
  var file = DriveApp.getFileById(templateID);
  var dest_folder = DriveApp.getFolderById(documentfolderID);

  var documentIDtest = checkFile(documentfolderID, filename);

  // If the file exists in the drive, return its ID. Else, create the file and return its ID.
  if(documentIDtest != "nothing here") {
    return documentIDtest;
  } else {
    // Create a copy of the file directly in the destination folder
    var copiedDocument = file.makeCopy(filename, dest_folder);

    // If needed, remove the original template file (though it's not clear why you'd want to remove the template)
    // DriveApp.getFolderById(templatefolderID).removeFile(file);

    return copiedDocument.getId();
  }
}



//------------------------APPEND FROM TEMPLATE TO LOGFILE------------------------------
//This function is used to append a template to an existing document.
//I.e., to append a new scan table or the Epilog to an existing Experiment Log.

function appendTemplate(templateID,documentID) {
  var templateDoc = DocumentApp.openById(templateID); //Pass in id of doc to be used as a template.
  var templateBody = templateDoc.getBody();

  var thisDoc = DocumentApp.openById(documentID);
  var thisBody = thisDoc.getBody();

  for(var i=0; i<templateBody.getNumChildren();i++){ //run through the elements of the template Body.
    switch (templateBody.getChild(i).getType()) { //Handle different elements to append.
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
  return "success"
}

function findAndReplace(documentID,keys,values) {
  var thisDoc = DocumentApp.openById(documentID);
  var thisBody = thisDoc.getBody();
  for(var i=0; i<keys.length;i++){
    thisBody.replaceText("(?i)"+"{{"+keys[i]+"}}", values[i])
  }
  return "(?i)"+"{{"+keys[1]+"}}"
}

//this is only for images in tables
function findAndReplaceImage(documentID,imageID,placeholder) {
 var thisDoc = DocumentApp.openById(documentID);
 var thisBody = thisDoc.getBody();
 var tables = thisBody.getTables();

 for (var k in tables)
 {
   var table = tables[k];
   var tablerows=table.getNumRows();
     for ( var row = 0; row < tablerows; ++row ) {
     var tablerow = table.getRow(row);
        for ( var cell=0; cell < tablerow.getNumCells(); ++cell) {
         var celltext = tablerow.getChild(cell).getText();

            if(celltext == placeholder) {
               var originalimage = DriveApp.getFileById(imageID).getBlob();
                table.replaceText("(?i)"+placeholder, "");
                table.getCell(row, cell).insertImage(0,originalimage);
            }
        }
     }
 }

return 0

}




//=================================HELPER FUNCTIONS=======================================

//CHECK FILE EXISTENCE IN A FOLDER
//Checks if the LogFile has been created before (or more generally, if file 'filename' exists in folder with folderID).
function checkFile(folderID, filename){
 var files = DriveApp.getFolderById(folderID).searchFiles('title contains "' + filename + '"');
  if(files.hasNext()==true){
    var file = files.next();
    var fileID = file.getId();
    return fileID}
  else{return "nothing here"}
}

//CHECKS STRING EXISTENCE WITHIN FILE
function checkFileContains(fileID, search){
  var searchresult = DocumentApp.openById(fileID).getBody().findText(search);
  if(searchresult!=null){
    return true}
  else{return false}
}


//CHECKS STRING EXISTENCE WITHIN FILE
function lastRowFromSpreadsheet(fileID, sheetString, firstCol,lastCol){
  var sheet = SpreadsheetApp.openById(fileID).getSheetByName(sheetString);

  var rng = sheet.getRange(firstCol+":"+lastCol).getValues();
  var lrIndex;

  for(var i = rng.length-1;i>=0;i--){
    lrIndex=i;
    if(!rng[i].every(function(c){ return c == ""; })){
      break;
    }

  }

  var lr = lrIndex+1;

  var rowVals = sheet.getRange(firstCol+lr+":"+lastCol+lr).getDisplayValues();

return rowVals;
}


function insertImageToTableCell(documentID, scanNumber, row, column, imageID) {
  var doc = DocumentApp.openById(documentID); // Open the document
  var body = doc.getBody(); // Get the document body
  var totalChildren = body.getNumChildren(); // Total elements in the body

  var headingRegex = new RegExp("^\\d{2}-\\d{2}-\\d{2} \\(\\d{2}:\\d{2}:\\d{2}\\) Scan " + scanNumber + "(:.*)?$"); // Regex for specific scan number
  var foundHeading = false; // Flag to indicate if the specific heading is found
  var tables = []; // Array to store tables under the found heading

  // Locate the heading and associated tables
  for (var i = 0; i < totalChildren; i++) {
    var child = body.getChild(i); // Get the current child element

    if (child.getType() === DocumentApp.ElementType.PARAGRAPH) {
      var paragraph = child.asParagraph();
      var text = paragraph.getText();

      if (paragraph.getHeading() === DocumentApp.ParagraphHeading.HEADING3 && headingRegex.test(text)) {
        foundHeading = true;

        // Look for tables after the found heading
        for (var j = i + 1; j < totalChildren; j++) {
          var nextChild = body.getChild(j);
          if (nextChild.getType() === DocumentApp.ElementType.TABLE) {
            tables.push(nextChild.asTable());
          } else if (nextChild.getType() === DocumentApp.ElementType.PARAGRAPH) {
            if (nextChild.asParagraph().getHeading() === DocumentApp.ParagraphHeading.HEADING3) {
              break;
            }
          }
        }
        break; // Exit the loop after finding the specific heading
      }
    }
  }

  // Insert the image into the specified table cell
  if (foundHeading && tables.length > 1) { // Ensure there is a second table
    var table = tables[1]; // Access the second table

    if (row < table.getNumRows() && column < table.getRow(row).getNumCells()) {
      var cell = table.getRow(row).getCell(column); // Access the specific cell
      cell.clear(); // Clear existing content if any

      // Get the image from Google Drive and insert it
      var imageBlob = DriveApp.getFileById(imageID).getBlob();
      cell.insertImage(0, imageBlob);
      Logger.log("Image inserted into Scan " + scanNumber + ", Row " + row + ", Column " + column);
    } else {
      Logger.log("Invalid row or column index for the table.");
    }
  } else if (foundHeading) {
    Logger.log("No second table found for Scan " + scanNumber);
  } else {
    Logger.log("Heading for Scan " + scanNumber + " not found.");
  }

  return 'success'
}


//=================================NEW FUNCTIONS (2025)=======================================

/**
 * Append a hyperlink paragraph at the end of the scan section for scanNumber.
 *
 * The paragraph is inserted just before the next Heading 3 (or at the end of
 * the document if this is the last scan entry). Multiple calls are safe and
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
  var sectionEnd = _sectionEndIndex(body, headingIndex);

  // Find the table cell containing "Additional diagnostics:" within this section.
  var diagCell = null;
  for (var i = headingIndex + 1; i < sectionEnd && !diagCell; i++) {
    var child = body.getChild(i);
    if (child.getType() !== DocumentApp.ElementType.TABLE) continue;
    var table = child.asTable();
    for (var r = 0; r < table.getNumRows() && !diagCell; r++) {
      var tableRow = table.getRow(r);
      for (var c = 0; c < tableRow.getNumCells() && !diagCell; c++) {
        var cell = tableRow.getCell(c);
        if (cell.getText().indexOf('Additional diagnostics') >= 0) {
          diagCell = cell;
        }
      }
    }
  }

  if (!diagCell) {
    Logger.log('"Additional diagnostics" cell not found in Scan ' + scanNumber + ' section.');
    return '"Additional diagnostics" cell not found.';
  }

  // Append a new hyperlink paragraph at the end of the cell.
  // Each successive call adds a new line, maintaining chronological order.
  var newPara = diagCell.insertParagraph(diagCell.getNumChildren(), label);
  newPara.editAsText().setLinkUrl(0, label.length - 1, url);
  Logger.log('Appended link for Scan ' + scanNumber + ': ' + label);
  return 'success';
}

// Private helpers shared by insertImageToTableCell and appendLinkToScan.

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
