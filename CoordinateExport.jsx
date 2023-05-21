//date: 2020-08-06
//author: mingfeng.zhao@castbox.fm
//function: rename exported file html to png.

var temp = "Lv";
var docPath = app.activeDocument.path;
var levelName = docPath.name;
var tempDir = String(docPath).replace(levelName, temp);
//init mix dir
var f = new Folder(docPath+"\\mix");
if ( ! f.exists ) {
	f.create()
}
//rename base
var baseHtmlFile = tempDir+"\\base.html";
var basePngFile = docPath+"\\base.png";
renameFile(baseHtmlFile,basePngFile);
//rename mix
var mixFolder = new Folder(tempDir+"\\mix");
var mixFiles = mixFolder.getFiles();
for(var i=0;i< mixFiles.length;i++){
	var NewName = String(mixFiles[i]).replace("html", "png").replace(temp,levelName);
	renameFile(String(mixFiles[i]),NewName)
}
function renameFile(oldFileName, newFileName){
	var file = new File(oldFileName);
	file.rename(newFileName);
	file.close();
}


//cleanup
//delete temp base file
var baseHtmlFile = tempDir+"\\base.html";
var basePngFile = tempDir+"\\base.png";
deleteFile(baseHtmlFile);
deleteFile(basePngFile);
//delete temp mix file
var mixFolder = new Folder(tempDir+"\\mix");
var mixFiles = mixFolder.getFiles();
for(var i=0;i< mixFiles.length;i++){
	deleteFile(String(mixFiles[i]))
}
function deleteFile(fileName){
	var MyFile = new File (fileName);
	if (MyFile.exists)
    {
        MyFile.remove();
     }
}

//delete temp mix dir
deleteDir(tempDir+"\\mix");
//delete temp dir
deleteDir(tempDir);

function deleteDir(dirName){
	var d = new Folder(dirName);
	if ( d.exists ) {
		d.remove();
	}
}