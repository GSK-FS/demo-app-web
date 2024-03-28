function previewImage(inputId, previewId, formId) {
  const input = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  const form = document.getElementById(formId);

  if (input.files && input.files[0]) {
    const reader = new FileReader();

    reader.onload = function (e) {
      preview.querySelector("img").src = e.target.result;
    };

    reader.readAsDataURL(input.files[0]);

    // Use AJAX to submit the form
    const formData = new FormData(form);
    const xhr = new XMLHttpRequest();

    xhr.open("POST", form.action, true);

    xhr.onreadystatechange = function () {
      if (xhr.readyState === 4 && xhr.status === 200) {
        console.log("File uploaded successfully");
        //print(`File uploaded successfully, ${inputId}`);
        // You can optionally handle success actions here
      }
    };

    xhr.send(formData);
  }
}

function openInNewTab(url) {
  // Enable the button before submitting the form
  document.getElementById("submitBtn").disabled = false;

  // Open a new tab with the specified URL
  window.open(url, "_blank");

  // Prevent the form from being submitted conventionally
  return false;
}
