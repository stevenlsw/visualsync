// Multi-Group Video Synchronization Script
// Add class names to videos to group them for synchronization
// Example: <video class="sync-group-1"> or <video class="sync-group-volleyball">
document.addEventListener('DOMContentLoaded', function() {
    // Configuration: Define which classes should be synchronized
    const syncGroups = [
        'sync-group-1',      // Example group 1
        'sync-group-2',      // Example group 2
        'sync-volleyball',   // Example: volleyball videos
        'sync-timeline',     // Example: timeline videos
        // Add more group classes as needed
    ];
    
    // Object to store video groups and their update flags
    const videoGroups = {};
    
    // Initialize groups
    syncGroups.forEach(groupClass => {
        const videos = document.querySelectorAll(`video.${groupClass}`);
        if (videos.length > 0) {
            videoGroups[groupClass] = {
                videos: Array.from(videos),
                isUpdating: false
            };
            console.log(`Found ${videos.length} videos in group: ${groupClass}`);
        }
    });
    
    // Also auto-detect groups based on any class starting with 'sync-'
    const allVideos = document.querySelectorAll('video[class*="sync-"]');
    allVideos.forEach(video => {
        const classes = video.className.split(' ');
        classes.forEach(className => {
            if (className.startsWith('sync-') && !videoGroups[className]) {
                const groupVideos = document.querySelectorAll(`video.${className}`);
                if (groupVideos.length > 0) {
                    videoGroups[className] = {
                        videos: Array.from(groupVideos),
                        isUpdating: false
                    };
                    console.log(`Auto-detected group: ${className} with ${groupVideos.length} videos`);
                }
            }
        });
    });
    
    // Function to find which group a video belongs to
    function getVideoGroup(targetVideo) {
        for (const [groupName, group] of Object.entries(videoGroups)) {
            if (group.videos.includes(targetVideo)) {
                return { groupName, group };
            }
        }
        return null;
    }
    
    // Function to sync videos within the same group
    function syncVideosInGroup(sourceVideo) {
        const videoGroup = getVideoGroup(sourceVideo);
        if (!videoGroup || videoGroup.group.isUpdating) return;
        
        videoGroup.group.isUpdating = true;
        
        videoGroup.group.videos.forEach(video => {
            if (video !== sourceVideo) {
                // Sync current time
                if (Math.abs(video.currentTime - sourceVideo.currentTime) > 0.1) {
                    video.currentTime = sourceVideo.currentTime;
                }
                
                // Sync play/pause state
                if (sourceVideo.paused && !video.paused) {
                    video.pause();
                } else if (!sourceVideo.paused && video.paused) {
                    video.play().catch(e => console.log('Play failed:', e));
                }
                
                // Sync volume and muted state
                video.volume = sourceVideo.volume;
                video.muted = sourceVideo.muted;
            }
        });
        
        setTimeout(() => {
            videoGroup.group.isUpdating = false;
        }, 50);
    }
    
    // Add event listeners to each video in all groups
    Object.entries(videoGroups).forEach(([groupName, group]) => {
        group.videos.forEach((video, index) => {
            // Play event
            video.addEventListener('play', function() {
                console.log(`Group ${groupName}, Video ${index + 1} started playing`);
                syncVideosInGroup(this);
            });
            
            // Pause event
            video.addEventListener('pause', function() {
                console.log(`Group ${groupName}, Video ${index + 1} paused`);
                syncVideosInGroup(this);
            });
            
            // Seeking event
            video.addEventListener('seeked', function() {
                console.log(`Group ${groupName}, Video ${index + 1} seeked to ${this.currentTime}s`);
                syncVideosInGroup(this);
            });
            
            // Time update for fine-grained sync
            let lastSyncTime = 0;
            video.addEventListener('timeupdate', function() {
                const now = Date.now();
                if (now - lastSyncTime > 500) {
                    syncVideosInGroup(this);
                    lastSyncTime = now;
                }
            });
            
            // Volume change event
            video.addEventListener('volumechange', function() {
                console.log(`Group ${groupName}, Video ${index + 1} volume changed`);
                syncVideosInGroup(this);
            });
            
            // Rate change (playback speed)
            video.addEventListener('ratechange', function() {
                console.log(`Group ${groupName}, Video ${index + 1} playback rate changed to ${this.playbackRate}x`);
                const videoGroup = getVideoGroup(this);
                if (!videoGroup || videoGroup.group.isUpdating) return;
                
                videoGroup.group.isUpdating = true;
                videoGroup.group.videos.forEach(otherVideo => {
                    if (otherVideo !== this) {
                        otherVideo.playbackRate = this.playbackRate;
                    }
                });
                
                setTimeout(() => {
                    videoGroup.group.isUpdating = false;
                }, 50);
            });
        });
    });
    
    // Keyboard shortcuts - works on the focused video group
    document.addEventListener('keydown', function(e) {
        if (e.target.tagName.toLowerCase() === 'input') return;
        
        // Find the most recently interacted video or the first video of the first group
        let targetVideo = document.activeElement?.tagName === 'VIDEO' ? document.activeElement : null;
        if (!targetVideo && Object.keys(videoGroups).length > 0) {
            const firstGroup = Object.values(videoGroups)[0];
            targetVideo = firstGroup.videos[0];
        }
        
        if (!targetVideo) return;
        
        switch(e.code) {
            case 'Space':
                e.preventDefault();
                if (targetVideo.paused) {
                    targetVideo.play();
                } else {
                    targetVideo.pause();
                }
                break;
                
            case 'ArrowLeft':
                e.preventDefault();
                targetVideo.currentTime = Math.max(0, targetVideo.currentTime - 5);
                break;
                
            case 'ArrowRight':
                e.preventDefault();
                targetVideo.currentTime = Math.min(targetVideo.duration, targetVideo.currentTime + 5);
                break;
                
            case 'KeyM':
                e.preventDefault();
                const videoGroup = getVideoGroup(targetVideo);
                if (videoGroup) {
                    const isMuted = targetVideo.muted;
                    videoGroup.group.videos.forEach(video => {
                        video.muted = !isMuted;
                    });
                }
                break;
        }
    });
    
    // Log initialization results
    const totalGroups = Object.keys(videoGroups).length;
    const totalVideos = Object.values(videoGroups).reduce((sum, group) => sum + group.videos.length, 0);
    
    console.log(`Video synchronization initialized!`);
    console.log(`- ${totalGroups} sync groups found`);
    console.log(`- ${totalVideos} total videos synchronized`);
    console.log('Groups:', Object.keys(videoGroups));
    console.log('\nKeyboard shortcuts (works on focused video group):');
    console.log('- Spacebar: Play/Pause group videos');
    console.log('- Left Arrow: Rewind 5 seconds');
    console.log('- Right Arrow: Forward 5 seconds');
    console.log('- M: Toggle mute for group');
    
    // Add visual indicator for grouped videos (optional)
    Object.entries(videoGroups).forEach(([groupName, group]) => {
        group.videos.forEach((video, index) => {
            // Add a subtle border to indicate grouping
            // video.style.border = `2px solid hsl(${groupName.length * 30}, 70%, 60%)`;
            // video.style.borderRadius = '4px';
            
            // Add title attribute for identification
            video.title = `Sync Group: ${groupName} (Video ${index + 1}/${group.videos.length})`;
        });
    });
});
