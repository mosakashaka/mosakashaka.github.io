function bootToc() {
    var elem0 = '<div class="ztree-div-left ztree" id="tree"></div>';
        elem1 = '<div class="ztree-div-right vscode-light" id="bootConentRoot"></div>';
    var o = new Array();
    document.body.childNodes.forEach(p => o.push(p.cloneNode(true)));
    document.body.innerHTML = "";
    document.body.append($(elem0)[0]);
    document.body.append($(elem1)[0]);

    var root = document.getElementById('bootConentRoot');
    o.forEach(p => root.appendChild(p));
}

$(document).ready(function(){
    bootToc();
    $('#tree').ztree_toc({

    });
});