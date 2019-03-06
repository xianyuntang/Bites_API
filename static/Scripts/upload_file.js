let prediction = function () {
    let data = new FormData();
    data.append('file', $('#file')[0].files[0]);
    $('#main').html('');
    $.ajax({
        type: "POST",
        url: 'upload_file',
        enctype: 'multipart/form-data',
        data: data,
        processData: false,
        contentType: false,
        success: function (result) {
            let text='';
            for (let i = 0; i < 8; i++) {
                text += '<p>'+result['label'][i]+'\t'+Math.round(result['result'][i],-3)+'%</p>'
            }
            $('#main').html(text);
        }
    })
};


$(document).ready(function () {
    $('#file').change(function () {
        document.getElementById('image').src = window.URL.createObjectURL(this.files[0])
        document.getElementById('image').style.display=''
    });

});