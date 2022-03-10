// 获得html元素
var radio_on = document.getElementById("radio_on")
var radio_off = document.getElementById("radio_off")
var video = document.getElementById("video");
var canvas = document.getElementById("canvas");
var ctx = canvas.getContext('2d');
var image = document.getElementById('receiver');
var canvas2 = document.getElementById("receiver_canvas")
var ctx2 = canvas2.getContext('2d')

// 获取摄像头视频流
function getMedia() {
    var constraints = {
        video: {width: 480, height: 320},
        audio: false
    };

    if(navigator.mediaDevices == undefined) {
        document.getElementById('reminder').innerHTML = '需使用https或者将地址添加到chrome://flags/#unsafely-treat-insecure-origin-as-secure'
    }

    var promise = navigator.mediaDevices.getUserMedia(constraints);
    promise.then(function (MediaStream) {
        video.srcObject = MediaStream;
        video.play();
    }).catch(function (PermissionDeniedError) {
        console.log(PermissionDeniedError);
    })
}

getMedia();

// 打开socket
var socket = new WebSocket("ws://162.105.89.56:35036/ws?video");

socket.onopen = function () {
    console.log("open success")
}

// 连接关闭的回调方法
socket.onclose = function () {
    console.log("close");
}

// 接收到消息的回调方法
socket.onmessage = function (data) {
    data = JSON.parse(data.data)
    image.src = data.img;

    data.df_prob = data.df_prob / 10
    option.series[0].data.shift()
    option.series[0].data.push(data.df_prob)
    chart.setOption(option)

    var face_tag = "正常"
    if (data.df_prob > 50) {
        face_tag = "可疑"
    }
    if(data._df_prob > 80) {
        face_tag = "异常"
    }

    if (data.df_prob > 80) {
        data.df_prob = 80;
    }
    else if (data.df_prob < 20) {
        data.df_prob = 20;
    }
    var hue = (80 - data.df_prob) * 2;

    ctx2.clearRect(0, 0, image.width, image.height)
    ctx2.fillStyle = "hsl(" + hue.toString() + ",100%, 50%)"
    ctx2.fillText(face_tag, data.rect[0]*canvas2.width/480, data.rect[1]*canvas2.height/320)
    ctx2.strokeStyle = "hsl(" + hue.toString() + ",100%, 50%)"
    ctx2.strokeRect(data.rect[0]*canvas2.width/480, data.rect[1]*canvas2.height/320, (data.rect[2]-data.rect[0])*canvas2.width/480, (data.rect[3]-data.rect[1])*canvas2.height/320)
}

// 初始化折线图
var chart = echarts.init(document.getElementById('chart'));
var x_data = []
var y_data = []
for (let i=1; i<=60; ++i) {
	x_data.push(i)
	y_data.push(0)
}
var option = {
                title: {
                    text: '伪造概率'
                },
                xAxis: {
					show: false,
                    position: "bottom",
                    type: "category",
					data: x_data
                },
                yAxis: {
                    show: true,
					name: "%",
                    position: "left",
                    type: "value",
                    min:0,
                    max:100,
                    splitNumber:5
                },
				series:[{
					symbol: 'none',
                    type: 'line',
                    data: y_data
                }]
            }
chart.setOption(option);

// 表示是否开启换脸
var is_radio_on = 'T'
radio_on.checked = true

function radio_check(status) {
    if (status == 0) {
        is_radio_on = 'T'
    }
    else {
        is_radio_on = 'F'
    }
}

// 设置定时任务向服务器发送图片
window.setInterval(function () {
    ctx.drawImage(video, 0, 0);
    socket.send(is_radio_on + canvas.toDataURL("image/jpeg", 0.85));
}, 60);