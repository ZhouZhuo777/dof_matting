//date: 2020-07-11
//author: mingfeng.zhao@castbox.fm
//function: generate slice center coordinate in visible layer.

var docPath = app.activeDocument.path;
var saveFileName = "coordinate";
var percision = 3;
var doc = app.activeDocument;
var layers = app.activeDocument.artLayers;
var outFile = docPath+"\\"+saveFileName;
saveFile(outFile, "w", doc.width.as("cm").toFixed(percision)+","+doc.height.as("cm").toFixed(percision));
var mix_frame_xy = new Array();
var index = 1
for(var i=0;i< layers.length;i++){
	if(layers[i].visible){
		// alert(layers[i].bounds);
		var bounds = layers[i].bounds;
		var x_a = bounds[0];
		var y_a = bounds[1];
		var x_b = bounds[2];
		var y_b = bounds[3];
		// alert(doc.width.as("cm").toFixed(percision));
		// alert(doc.height.as("cm").toFixed(percision));
		if ( x_b > doc.width){
			// alert("x超出");
			x_b = doc.width
		}
		if ( y_b > doc.height){
			// alert("y超出");
			y_b = doc.height
		}
		var x_o = x_a + (x_b - x_a)/2;
		var y_o = y_a + (y_b - y_a)/2;
		mix_frame_xy[layers[i].name] = x_o.as("cm").toFixed(percision)+","+y_o.as("cm").toFixed(percision)
		// alert(index+":"+x_o.as("cm").toFixed(percision)+","+y_o.as("cm").toFixed(percision));
		// saveFile(outFile, "a", layers[i].name+":"+x_o.as("cm").toFixed(percision)+","+y_o.as("cm").toFixed(percision));
		index++
	}
}

for(var j=1; j<=index/2; j++){
	var frame = j+''+j;
	saveFile(outFile, "a", j+":"+mix_frame_xy[j]+":"+mix_frame_xy[frame]);
}

function saveFile(filePath, type, content){
	var file = new File(filePath);
	file.open(type);
	file.writeln(content);
	file.close();
}