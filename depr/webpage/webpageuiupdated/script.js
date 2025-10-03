document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.getElementById('sidebar');
    const hamburgerIcon = document.getElementById('hamburger-icon');
    const mapLink = document.getElementById('map-link');
    const dashboardLink = document.getElementById('dashboard-link');
    const mapView = document.getElementById('map-view');
    const dashboardView = document.getElementById('dashboard-view');
    // const body = document.body; // No longer strictly needed for body class toggle

    let map; // Declare map variable globally or in a scope accessible to initMap

    // Function to initialize the map
    function initMap() {
        if (map) {
            map.remove(); // Remove existing map if it exists
        }
        map = L.map('mapid').setView([40, -100], 3); // Centered on North America, zoom level 3

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);
    }

    // Toggle sidebar visibility
    hamburgerIcon.addEventListener('click', function() {
        sidebar.classList.toggle('hidden');
        // Invalidate map size when sidebar toggles to ensure it renders correctly
        if (map && mapView.classList.contains('active')) { // Only invalidate if map is currently active
            setTimeout(() => {
                map.invalidateSize();
            }, 300); // Give CSS transition time to complete (should match CSS transition duration)
        }
    });

    // Navigation links
    mapLink.addEventListener('click', function(e) {
        e.preventDefault();
        mapLink.classList.add('active');
        dashboardLink.classList.remove('active');
        mapView.classList.add('active');
        dashboardView.classList.remove('active');
        initMap(); // Re-initialize map when navigating to map view
        // Ensure map size is correct after switching to map view
        if (map) {
            setTimeout(() => {
                map.invalidateSize();
            }, 0); // Invalidate immediately after view switch
        }
    });

    dashboardLink.addEventListener('click', function(e) {
        e.preventDefault();
        dashboardLink.classList.add('active');
        mapLink.classList.remove('active');
        dashboardView.classList.add('active');
        mapView.classList.remove('active');
        // No need to destroy map, just hide its container
    });

    // Initialize the map on page load for the default "Map View"
    initMap();
});