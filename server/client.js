//获得video摄像头区域
  var video = document.getElementById("video");
  var canvas = document.getElementById("canvas");
  var ctx = canvas.getContext('2d');
  var image = document.getElementById('receiver');


  function getMedia() {
    var constraints = {
      video: {width: 480, height: 320},
      audio: false
    };

    var promise = navigator.mediaDevices.getUserMedia(constraints);
    promise.then(function (MediaStream) {
      video.srcObject = MediaStream;
      video.play();
    }).catch(function (PermissionDeniedError) {
      console.log(PermissionDeniedError);
    })
  }

  //打开socket
  var socket = new WebSocket("ws://162.105.89.56:35036/ws?video");

  socket.onopen = function () {
    console.log("open success")
  }

  //连接关闭的回调方法
  socket.onclose = function () {
    console.log("close");
  }

  //接收到消息的回调方法
  socket.onmessage = function (data) {
    image.src = data.data;
  }

  getMedia();

  var interval = window.setInterval(function () {
    ctx.drawImage(video, 0, 0);
    socket.send(canvas.toDataURL("image/jpeg", 0.85));
    console.log(cnt)
  }, 60);
