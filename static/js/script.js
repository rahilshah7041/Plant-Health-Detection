
document.addEventListener('DOMContentLoaded', function() {

document.getElementById('analysis-form').addEventListener('submit', function (e) {
    e.preventDefault();

    var formData = new FormData(this);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        document.getElementById('result').innerHTML = 'Prediction is: ' + data.predicted_class;
    })
    .catch(error => console.error('Error:', error));
});

});