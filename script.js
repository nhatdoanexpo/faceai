async function loadModels() {
  const MODEL_URL = '/models';
  await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
  await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
}

async function detectAllLabeledFaces() {
  const labels = ["Nancy", "Yeonwoo" ];
  const labeledFaceDescriptors = await Promise.all(
      labels.map(async label => {
        const descriptions = [];
        for (let i = 1; i <= 2; i++) {
          const img = await faceapi.fetchImage(`/data/${label}/${i}.jpg`);
          const detection = await faceapi
              .detectSingleFace(img)
              .withFaceLandmarks()
              .withFaceDescriptor()
              .withFaceExpressions()
          ;
          descriptions.push(detection.descriptor);
        }
        return new faceapi.LabeledFaceDescriptors(label, descriptions);
      })
  );

  // Lưu đối tượng LabeledFaceDescriptors vào localStorage
  const json = JSON.stringify(labeledFaceDescriptors.map(fd => ({
    label: fd.label,
    descriptors: fd.descriptors.map(d => Array.from(d))
  })));
  localStorage.setItem('labeledFaceDescriptors', json);

  return labeledFaceDescriptors;
}

function loadLabeledFaceDescriptorsFromLocalStorage() {
  const json = localStorage.getItem('labeledFaceDescriptors');
  if (!json) return null;

  const parsed = JSON.parse(json);
  return parsed.map(fd => new faceapi.LabeledFaceDescriptors(
      fd.label,
      fd.descriptors.map(d => new Float32Array(d))
  ));
}

document.addEventListener('DOMContentLoaded', async () => {
  await loadModels();

  // Lắng nghe sự kiện khi người dùng chọn file
  document.getElementById('imageUpload1').addEventListener('change', function(event) {
    // Lấy đối tượng file đã chọn từ input
    var file = event.target.files[0];

    // Tạo đường dẫn URL cho hình ảnh đã chọn
    var imgUrl = URL.createObjectURL(file);

    // Đặt đường dẫn mới cho thuộc tính src của <img>
    var imgElement = document.getElementById('myImg');
    imgElement.src = imgUrl;
  });


  const imageUpload1 = document.getElementById('imageUpload1');
  const detectButton = document.getElementById('detect');

  const canvas = document.getElementById('myCanvas');
  const ctx = canvas.getContext('2d');

  detectButton.addEventListener('click', async () => {
    if (imageUpload1.files.length === 0) {
      alert('Vui lòng tải lên một hình ảnh.');
      return;
    }
    // Xóa bất kỳ khung nào đã vẽ trước đó trên canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Detect tim Face tren anh
    const result = await faceapi
        .detectSingleFace(myImg, new faceapi.SsdMobilenetv1Options())
        .withFaceLandmarks()
        .withFaceDescriptor()
        .withFaceExpressions()
    ;
     // nếu tìm thấy  có mặt người trong ảnh thì làm j đó
    if (result) {
      // Kiểm tra cảm xúc của khuôn mặt
      const expressions = result.expressions;
      let emotion = 'Không xác định'; // Giả sử mặc định là không xác định

      if (expressions) {
        // Tìm biểu hiện cảm xúc có giá trị lớn nhất
        const maxValue = Math.max(...Object.values(expressions));
        emotion = Object.keys(expressions).find(key => expressions[key] === maxValue);
      }

      // // Vẽ khung xung quanh khuôn mặt nhận diện được
      const dims = faceapi.matchDimensions(canvas, myImg, true);
      const resizedResult = faceapi.resizeResults(result, dims);
      ctx.drawImage(myImg, 0, 0, dims.width, dims.height);
      // faceapi.draw.drawDetections(canvas, resizedResult);

      // Load Labeled Face Descriptors from Local Storage
      //lấy data đã trainning khuôn mặt
      let labeledFaceDescriptors = loadLabeledFaceDescriptorsFromLocalStorage();
      if (!labeledFaceDescriptors) {
        // If not found in Local Storage, detect and save them
        labeledFaceDescriptors = await detectAllLabeledFaces();
      }
       // tìm khuôn mặt match với data đã trainning
      const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
      const bestMatch = faceMatcher.findBestMatch(result.descriptor);
      const box = resizedResult.detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, { label: bestMatch.label +"-----cảm xúc: "+emotion });
      drawBox.draw(canvas);
      document.getElementById('result').innerText = `Kết quả: ${bestMatch.label} cảm xúc: ${emotion} `;
    } else {
      document.getElementById('result').innerText = 'Không thể nhận diện khuôn mặt.';
    }
  });
});
