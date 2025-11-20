// Common JavaScript functions for all pages
// Scene and Method Management

let currentScene = 0;
let currentMethod = 0;
let scenes = [];
let methods = [];
let syncSelectors = {};

// Initialize with configuration
function initPageConfig(config) {
  scenes = config.scenes || [];
  methods = config.methods || [];
  syncSelectors = config.syncSelectors || {};
  
  // Set initial view
  window.onload = updateView;
}

// Video synchronization functions
function syncVideosInGroup(videoSelector) {
  let isUpdating = false;
  const videos = document.querySelectorAll(videoSelector);

  videos.forEach((video, index) => {
    // Store handlers so we can remove them later
    const handlers = {
      play: function() {
        if (isUpdating) return;
        isUpdating = true;
        videos.forEach(v => {
          if (v !== video && v.paused) {
            v.play().catch(err => console.log('Play failed:', err));
          }
        });
        setTimeout(() => isUpdating = false, 50);
      },
      pause: function() {
        if (isUpdating) return;
        isUpdating = true;
        videos.forEach(v => {
          if (v !== video && !v.paused) {
            v.pause();
          }
        });
        setTimeout(() => isUpdating = false, 50);
      },
      seeked: function() {
        if (isUpdating) return;
        isUpdating = true;
        const targetTime = video.currentTime;
        videos.forEach(v => {
          if (v !== video && Math.abs(v.currentTime - targetTime) > 0.1) {
            v.currentTime = targetTime;
          }
        });
        setTimeout(() => isUpdating = false, 50);
      },
      timeupdate: function() {
        if (isUpdating) return;
        const targetTime = video.currentTime;
        videos.forEach(v => {
          if (v !== video && Math.abs(v.currentTime - targetTime) > 0.3) {
            isUpdating = true;
            v.currentTime = targetTime;
            setTimeout(() => isUpdating = false, 50);
          }
        });
      }
    };

    video._syncHandlers = handlers;
    video.addEventListener('play', handlers.play);
    video.addEventListener('pause', handlers.pause);
    video.addEventListener('seeked', handlers.seeked);
    video.addEventListener('timeupdate', handlers.timeupdate);
  });
}

function clearAllVideoListeners() {
  document.querySelectorAll('video').forEach(video => {
    if (video._syncHandlers) {
      video.removeEventListener('play', video._syncHandlers.play);
      video.removeEventListener('pause', video._syncHandlers.pause);
      video.removeEventListener('seeked', video._syncHandlers.seeked);
      video.removeEventListener('timeupdate', video._syncHandlers.timeupdate);
      delete video._syncHandlers;
    }
  });
}

function updateView() {
  // Clear existing video synchronization
  clearAllVideoListeners();

  // Hide all videos
  document.querySelectorAll('.video-container, .timeline-container').forEach(container => {
    container.style.display = 'none';
  });

  // Show current scene and method
  const sceneClass = scenes[currentScene] + '-video';
  const methodClass = methods[currentMethod] + '-video';
  
  document.querySelectorAll('.' + sceneClass + '.' + methodClass).forEach(container => {
    container.style.display = 'block';
  });

  // Update active tab styles
  document.querySelectorAll('#scene-selector li').forEach((li, idx) =>
    li.classList.toggle('active', idx === currentScene)
  );
  document.querySelectorAll('#method-selector li').forEach((li, idx) =>
    li.classList.toggle('active', idx === currentMethod)
  );

  // Set up video synchronization for the currently visible videos
  setTimeout(() => {
    const currentSceneName = scenes[currentScene];
    const currentMethodName = methods[currentMethod];
    
    const selector = syncSelectors[currentSceneName]?.[currentMethodName];
    if (selector) {
      syncVideosInGroup(selector);
    }
  }, 150);
}

function ChangeScene(index) {
  currentScene = index;
  updateView();
}

function ChangeMethod(index) {
  currentMethod = index;
  updateView();
}

// Add colored borders to video containers
document.addEventListener('DOMContentLoaded', function() {
  const videoContainers = document.querySelectorAll('.video-container');
  const colors = ['#3498DB', '#2ECC71', '#F39C12']; // Blue, Green, Orange

  videoContainers.forEach((container, index) => {
    const colorIndex = index % colors.length;
    container.style.borderColor = colors[colorIndex];
  });
});



