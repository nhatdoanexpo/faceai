async function loadModels() {
  const MODEL_URL = '/models';
  await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
  await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
}

async function detectAllLabeledFaces() {
  const labels = ["Nancy", "Yeonwoo","Chi" ];
  const labeledFaceDescriptors = await Promise.all(
      labels.map(async label => {
        const descriptions = [];
        for (let i = 1; i <= 4; i++) {
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
//----------------------------
// Lấy đối tượng input imageUpload từ HTML
async function trainNewImage() {
  const inputElement = document.getElementById('imageUpload');
  const file = inputElement.files[0];

  if (!file) {
    alert('Vui lòng chọn một file ảnh.');
    return;
  }

  try {
    // Tải hình ảnh và nhận diện khuôn mặt
    const img = await faceapi.bufferToImage(file);
    const detection = await faceapi.detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor()
        .withFaceExpressions();

    if (!detection) {
      alert('Không tìm thấy khuôn mặt trong ảnh.');
      return;
    }

    // Nhãn là tên người trong ảnh (có thể lấy từ người dùng nhập hoặc từ một nguồn khác)
    const label = prompt('Nhập tên của người trong ảnh:');
    if (!label) return;

    // Tạo đối tượng LabeledFaceDescriptors mới
    const labeledFaceDescriptor = new faceapi.LabeledFaceDescriptors(
        label,
        [detection.descriptor]
    );

    // Lấy danh sách các LabeledFaceDescriptors từ localStorage
    let labeledFaceDescriptors = loadLabeledFaceDescriptorsFromLocalStorage() || [];

    // Thêm labeledFaceDescriptor mới vào mảng
    labeledFaceDescriptors.push(labeledFaceDescriptor);

    // Lưu lại vào localStorage
    const json = JSON.stringify(labeledFaceDescriptors.map(fd => ({
      label: fd.label,
      descriptors: fd.descriptors.map(d => Array.from(d))
    })));
    localStorage.setItem('labeledFaceDescriptors', json);

    alert('Đã thêm ảnh và huấn luyện thành công.');
  } catch (error) {
    console.error('Lỗi khi huấn luyện ảnh:', error);
    alert('Đã xảy ra lỗi khi huấn luyện ảnh.');
  }
}
//------------------------- huon luyen tu cammera
// Hàm để huấn luyện ảnh từ camera
async function trainFromCamera() {
  try {
    // Lấy video từ camera
    const videoElement = document.createElement('video');
    document.body.appendChild(videoElement);
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = stream;
    await videoElement.play();

    // Chờ một chút để camera hiển thị và người dùng chọn góc nhìn
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Tạo canvas để hiển thị ảnh từ camera
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');

    // Vẽ hình ảnh từ video lên canvas
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    // Lấy ảnh từ canvas để nhận diện
    const img = canvas;

    // Thực hiện nhận diện khuôn mặt và các bước khác
    const detection = await faceapi.detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor()
        .withFaceExpressions();

    if (!detection) {
      alert('Không tìm thấy khuôn mặt trong ảnh từ camera.');
      return;
    }

    // Nhãn là tên người trong ảnh (có thể lấy từ người dùng nhập hoặc từ một nguồn khác)
    const label = prompt('Nhập tên của người trong ảnh:');
    if (!label) return;

    // Tạo đối tượng LabeledFaceDescriptors mới
    const labeledFaceDescriptor = new faceapi.LabeledFaceDescriptors(
        label,
        [detection.descriptor]
    );

    // Lấy danh sách các LabeledFaceDescriptors từ localStorage
    let labeledFaceDescriptors = loadLabeledFaceDescriptorsFromLocalStorage() || [];

    // Thêm labeledFaceDescriptor mới vào mảng
    labeledFaceDescriptors.push(labeledFaceDescriptor);

    // Lưu lại vào localStorage
    const json = JSON.stringify(labeledFaceDescriptors.map(fd => ({
      label: fd.label,
      descriptors: fd.descriptors.map(d => Array.from(d))
    })));
    localStorage.setItem('labeledFaceDescriptors', json);

    alert('Đã thêm ảnh từ camera và huấn luyện thành công.');

    // Dừng và giải phóng stream từ camera
    stream.getVideoTracks().forEach(track => track.stop());
    document.body.removeChild(videoElement);
  } catch (error) {
    console.error('Lỗi khi huấn luyện ảnh từ camera:', error);
    alert('Đã xảy ra lỗi khi huấn luyện ảnh từ camera.');
  }
}


/////--------------------------





document.addEventListener('DOMContentLoaded', async () => {
  await loadModels();

  // Lắng nghe sự kiện khi người dùng chọn file để huấn luyện ảnh mới
  document.getElementById('trainButton').addEventListener('click', async () => {
    await trainNewImage();
  });

  const trainCameraButton = document.getElementById('trainCameraButton');
  trainCameraButton.addEventListener('click', async () => {
    await trainFromCamera();
  });

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

    // Detect Face
    const result = await faceapi
        .detectSingleFace(myImg, new faceapi.SsdMobilenetv1Options())
        .withFaceLandmarks()
        .withFaceDescriptor()
        .withFaceExpressions()
    ;

    if (result) {
      // // Vẽ khung xung quanh khuôn mặt nhận diện được
      const dims = faceapi.matchDimensions(canvas, myImg, true);
      const resizedResult = faceapi.resizeResults(result, dims);
      ctx.drawImage(myImg, 0, 0, dims.width, dims.height);
      // faceapi.draw.drawDetections(canvas, resizedResult);

      // Load Labeled Face Descriptors from Local Storage
      let labeledFaceDescriptors = loadLabeledFaceDescriptorsFromLocalStorage();
      if (!labeledFaceDescriptors) {
        // If not found in Local Storage, detect and save them
        labeledFaceDescriptors = await detectAllLabeledFaces();
      }
      // Kiểm tra cảm xúc của khuôn mặt
      const expressions = result.expressions;
      let emotion = 'Không xác định'; // Giả sử mặc định là không xác định

      if (expressions) {
        // Tìm biểu hiện cảm xúc có giá trị lớn nhất
        const maxValue = Math.max(...Object.values(expressions));
        emotion = Object.keys(expressions).find(key => expressions[key] === maxValue);
      }
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


async function detectContinuousFromCamera() {
  try {
    const videoElement = document.createElement('video');
    document.body.appendChild(videoElement);
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = stream;
    await videoElement.play();

    // Chờ một chút để camera hiển thị và người dùng chọn góc nhìn
    await new Promise(resolve => setTimeout(resolve, 2000));

    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');

    // Loop để liên tục nhận diện từ camera
    setInterval(async () => {
      ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
      const img = canvas;

      const detection = await faceapi.detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor()
          .withFaceExpressions();

      if (detection) {
        // Nhận diện thành công
        const label = detection.descriptor;

        // Load Labeled Face Descriptors from Local Storage
        let labeledFaceDescriptors = loadLabeledFaceDescriptorsFromLocalStorage();
        if (!labeledFaceDescriptors) {
          // Nếu không tìm thấy trong Local Storage, detect và lưu chúng
          labeledFaceDescriptors = await detectAllLabeledFaces();
        }

        // Nhận diện người gần nhất dựa trên descriptors
        const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.4);
        const bestMatch = faceMatcher.findBestMatch(label);

        // Lấy cảm xúc của khuôn mặt
        const expressions = detection.expressions;
        let emotion = 'Không xác định'; // Giả sử mặc định là không xác định

        if (expressions) {
          // Tìm biểu hiện cảm xúc có giá trị lớn nhất
          const maxValue = Math.max(...Object.values(expressions));
          emotion = Object.keys(expressions).find(key => expressions[key] === maxValue);
        }

        // Vẽ khung và hiển thị thông tin
        const dims = faceapi.matchDimensions(canvas, img, true);
        const resizedResult = faceapi.resizeResults(detection, dims);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, dims.width, dims.height);
        const box = resizedResult.detection.box;
        const drawBox = new faceapi.draw.DrawBox(box, { label: bestMatch.label + " - Cảm xúc: " + emotion });
        drawBox.draw(canvas);

        document.getElementById('result').innerText = `Kết quả: ${bestMatch.label} - Cảm xúc: ${emotion}`;
      } else {
        // Không nhận diện được khuôn mặt
        document.getElementById('result').innerText = 'Không thể nhận diện khuôn mặt.';
      }
    }, 1000); // Thực hiện nhận diện mỗi giây

  } catch (error) {
    console.error('Lỗi khi huấn luyện ảnh từ camera:', error);
    alert('Đã xảy ra lỗi khi nhận diện từ camera.');
  }
}