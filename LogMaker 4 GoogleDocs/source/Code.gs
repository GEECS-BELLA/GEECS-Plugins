//HTT EXPERIMENT LOG GENERATOR APPS Script
//Needs to be installed as a google apps script and deployed as API

/**
 * This project works with a local python interface calling the functions via Google Apps Scripts API.
 * The purpose of this script is to make automated experiment logs at BELLA Center.
 * 
 * by Tobias Ostermayr, last updated March 31, 2020
 */

//=================================MAIN FUNCTIONS======================================

//--------------------------------LOG FILE CREATOR-------------------------------------
//If no Logfile has been created that day, this function copies the template logfile into the Experiment Log folder and returns its ID.
//If the Logfile does exist already, this function returns its ID
function createExperimentLog(templateID,templatefolderID,documentfolderID,filename) {
  var date = Utilities.formatDate(new Date(), 'America/Los_Angeles', "MM-dd-YY"); 
  var file = DriveApp.getFileById(templateID);
  var source_folder = DriveApp.getFolderById(templatefolderID);
  var dest_folder = DriveApp.getFolderById(documentfolderID);
  var documentIDtest = checkFile(documentfolderID, filename);
  //If the file exists in the drive return its ID. Else, create the file and return its ID.
  if(documentIDtest != "nothing here"){
  return documentIDtest
  }else{
  var file2 = file.makeCopy(filename);
  dest_folder.addFile(file2);
  source_folder.removeFile(file2);
  var documentID = file2.getId();
  return documentID
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
            //var tablewidth = tablerow.getChild(cell).getWidth();
            //tablewidth = tablewidth >> 0;
            //var imagesize = ImgApp.getSize(originalimage);
              
            //var width = imagesize.width;
              
            //  if (width > tablewidth) {
              //  var res = ImgApp.doResize(imageID, tablewidth);
              //  table.replaceText("(?i)"+placeholder, "");
            //var tablewidth = table.getCell(row,cell).getWidth();
              //  table.getCell(row, cell).insertImage(0,res.blob);}
              //else{
                table.replaceText("(?i)"+placeholder, "");
                table.getCell(row, cell).insertImage(0,originalimage);
                //}
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