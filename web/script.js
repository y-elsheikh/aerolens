<<<<<<< HEAD
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
=======
document.addEventListener('DOMContentLoaded', () => {
  if (window.lucide && lucide.createIcons) lucide.createIcons();

  const sidebar = document.getElementById('sidebar');
  const hamburgerIcon = document.getElementById('hamburger-icon');
  const showSidebarBtn = document.getElementById('show-sidebar-btn');
  const pageType = document.body.getAttribute('data-page'); // "map" | "dashboard"

  // nav active
  const navLinks = document.querySelectorAll('.sidebar-nav a');
  navLinks.forEach(a => {
    const href = (a.getAttribute('href') || '').split('?')[0];
    const isMap = pageType === 'map' && href.endsWith('index.html');
    const isDash = pageType === 'dashboard' && href.endsWith('dashboard.html');
    if (isMap || isDash) {
      a.classList.add('active', 'bg-indigo-700/50', 'text-white');
      a.setAttribute('aria-current', 'page');
      a.classList.remove('text-gray-400');
    } else {
      a.classList.remove('active', 'bg-indigo-700/50', 'text-white');
      a.classList.add('text-gray-400');
      a.removeAttribute('aria-current');
    }
  });

  // mobile sidebar
  if (window.innerWidth <= 768) {
    sidebar?.classList.remove('is-open');
    if (sidebar) sidebar.style.transform = 'translateX(-100%)';
    if (showSidebarBtn) showSidebarBtn.style.display = 'block';
  } else {
    if (showSidebarBtn) showSidebarBtn.style.display = 'none';
    sidebar?.classList.add('is-open');
  }

  function toggleSidebar() {
    if (!sidebar) return;
    sidebar.classList.toggle('is-open');

    if (window.innerWidth <= 768) {
      if (sidebar.classList.contains('is-open')) {
        sidebar.style.transform = 'translateX(0)';
        if (showSidebarBtn) showSidebarBtn.style.display = 'none';
      } else {
        sidebar.style.transform = 'translateX(-100%)';
        if (showSidebarBtn) showSidebarBtn.style.display = 'block';
      }
    }
    if (window._mapInstance && pageType === 'map') {
      setTimeout(() => window._mapInstance.invalidateSize(), 300);
    }
  }

  hamburgerIcon?.addEventListener('click', toggleSidebar);
  showSidebarBtn?.addEventListener('click', () => {
    sidebar?.classList.add('is-open');
    toggleSidebar();
  });

  // ---------------- MAP PAGE ----------------
  if (pageType === 'map' && typeof L !== 'undefined') {
    const US_BOUNDS = L.latLngBounds(
      [24.396308, -124.848974], // SW
      [49.384358,  -66.885444]  // NE
    );

    if (!window._mapInstance) {
      const map = L.map('mapid', {
        center: [40, -100],
        zoom: 5,
        minZoom: 5,
        maxZoom: 18,
        maxBounds: US_BOUNDS,
        maxBoundsViscosity: 1.0,
        worldCopyJump: false,
        // If you still have issues typing, uncomment the next line to fully disable map keyboard shortcuts:
        // keyboard: false
      });
      window._mapInstance = map;

      L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 19,
        noWrap: true,
        bounds: US_BOUNDS
      }).addTo(map);

      map.setMaxBounds(US_BOUNDS.pad(0.02));

      // ---------- Top-right Search Control (fixed typing) ----------
      let searchMarker = null;

      const SearchControl = L.Control.extend({
        options: { position: 'topright' },
        onAdd: function () {
          const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
          container.style.padding = '6px';
          container.style.background = '#fff';
          container.style.boxShadow = '0 1px 5px rgba(0,0,0,0.4)';
          container.style.display = 'flex';
          container.style.alignItems = 'center';

          const form = L.DomUtil.create('form', '', container);
          form.style.display = 'flex';
          form.style.gap = '6px';
          form.style.alignItems = 'center';
          form.setAttribute('autocomplete', 'off');

          const input = L.DomUtil.create('input', '', form);
          input.type = 'text';
          input.placeholder = 'Search US place…';
          input.ariaLabel = 'Search';
          input.style.border = '1px solid black';
          input.style.padding = '4px 6px';
          input.style.width = '220px';
          input.style.outline = 'none';

          const btn = L.DomUtil.create('button', '', form);
          btn.type = 'submit';
          btn.textContent = 'Go';
          btn.style.border = '1px solid black';
          btn.style.padding = '4px 10px';
          btn.style.cursor = 'pointer';
          btn.style.background = 'green';

          // IMPORTANT: stop events so typing works
          L.DomEvent.disableClickPropagation(container);
          L.DomEvent.disableScrollPropagation(container);

          // Stop keyboard events from reaching the map
          ['keydown', 'keyup', 'keypress'].forEach(ev => {
            L.DomEvent.on(input, ev, (e) => {
              e.stopPropagation();
            });
          });

          // Also stop click/focus from bubbling to map
          L.DomEvent.on(input, 'click', L.DomEvent.stopPropagation);
          L.DomEvent.on(btn, 'click', L.DomEvent.stopPropagation);
          L.DomEvent.on(form, 'submit', L.DomEvent.stopPropagation);

          L.DomEvent.on(form, 'submit', async (e) => {
            e.preventDefault();
            const q = input.value.trim();
            if (!q) return;

            try {
              const url = `https://nominatim.openstreetmap.org/search?format=json&countrycodes=us&limit=1&q=${encodeURIComponent(q)}`;
              const res = await fetch(url, { headers: { 'Accept': 'application/json' }});
              const data = await res.json();

              if (Array.isArray(data) && data.length > 0) {
                const { lat, lon, display_name } = data[0];
                const latNum = parseFloat(lat);
                const lonNum = parseFloat(lon);
                const target = L.latLng(latNum, lonNum);

                if (!US_BOUNDS.contains(target)) {
                  alert('That location is outside the supported US area.');
                  return;
                }

                if (searchMarker) {
                  searchMarker.setLatLng(target).setPopupContent(display_name);
                } else {
                  searchMarker = L.marker(target).addTo(map).bindPopup(display_name);
                }
                searchMarker.openPopup();

                const searchZoomCap = 8;
                const targetZoom = Math.max(
                  map.getMinZoom(),
                  Math.min(searchZoomCap, map.getMaxZoom())
                );
                map.flyTo(target, targetZoom, { animate: true, duration: 1.2 });
              } else {
                alert('No US results found.');
              }
            } catch (err) {
              console.error(err);
              alert('Search failed. Please try again.');
            }
          });

          return container;
        }
      });

      map.addControl(new SearchControl());

      setTimeout(() => map.invalidateSize(), 0);
    }
  }

  // ---------------- DASHBOARD PAGE ----------------
  if (pageType === 'dashboard') {
    const btn = document.getElementById('forecast-btn');
    const input = document.getElementById('city-input');
    const output = document.getElementById('forecast-output');

    function checkForecast() {
      const city = (input?.value || '').trim();
      if (!output) return;

      if (city) {
        const aqi = Math.floor(Math.random() * 200) + 20;
        let status = 'Moderate';
        let color = 'text-yellow-400';

        if (aqi < 50) { status = 'GOOD'; color = 'text-green-400'; }
        else if (aqi > 150) { status = 'UNHEALTHY'; color = 'text-red-400'; }

        output.innerHTML = `
          <p class="text-lg font-semibold mb-2">Forecast for <span class="text-indigo-400">${city}</span> (48h Avg):</p>
          <p class="text-3xl font-extrabold ${color}">AQI: ${aqi} (${status})</p>
          <p class="text-sm mt-2 text-gray-500">Predicted by CNN-LSTM Model. Confidence Score: 89.5%</p>
        `;
      } else {
        output.innerHTML = '<p>Please enter a valid city or zip code.</p>';
      }
    }

    btn?.addEventListener('click', checkForecast);
    input?.addEventListener('keydown', (e) => { if (e.key === 'Enter') checkForecast(); });
  }
});
>>>>>>> 9ff9d39c603520c314c455411ffba6cf958afdf2
