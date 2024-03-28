function processImages() {
  var form = document.getElementById("upload-form");
  var formData = new FormData(form);

  fetch("/process", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      displayImages(data);
    });
}

function displayImages(data) {
  var outputContainer = document.getElementById("output-container");
  outputContainer.innerHTML = "";

  for (var key in data) {
    if (data.hasOwnProperty(key)) {
      var img = document.createElement("img");
      img.src = "data:image/jpeg;base64," + data[key];
      outputContainer.appendChild(img);
    }
  }
}
