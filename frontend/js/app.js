function toggleMapOptions() {
    const mapOptions = document.getElementById('map-options');
    if (document.getElementById('generate-map').checked) {
        mapOptions.style.display = 'block';
    } else {
        mapOptions.style.display = 'none';
    }
}