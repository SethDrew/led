// LED Effects Gallery - Interactive Controls

// Effect data structure
const effectsData = {
    backgrounds: {
        'NebulaBackground': {
            name: 'Nebula',
            description: 'Breathing waves with color shifts',
            className: 'NebulaBackground'
        },
        'SolidColorBackground': {
            name: 'Solid Color',
            description: 'Static solid color background',
            className: 'SolidColorBackground'
        },
        'PulsingColorBackground': {
            name: 'Pulsing Color',
            description: 'Pulsing solid color effect',
            className: 'PulsingColorBackground'
        }
    },
    foregrounds: {
        'CrawlingStarsForeground': {
            name: 'Crawling Stars',
            description: 'Glowing orbs that drift along the strip',
            className: 'CrawlingStarsForeground'
        },
        'SparksForeground': {
            name: 'Sparks',
            description: 'Random spark explosions',
            className: 'SparksForeground'
        },
        'CollisionForeground': {
            name: 'Collision',
            description: 'Crawlers that collide and explode',
            className: 'CollisionForeground'
        },
        'RainbowCircleForeground': {
            name: 'Rainbow Circle',
            description: 'Rainbow circle passing through',
            className: 'RainbowCircleForeground'
        },
        'EnhancedCrawlForeground': {
            name: 'Enhanced Crawl',
            description: 'Smooth wave with color modes',
            className: 'EnhancedCrawlForeground'
        },
        'FragmentationForeground': {
            name: 'Fragmentation',
            description: 'White crawler decomposes to RGB fragments',
            className: 'FragmentationForeground'
        },
        'DriftingDecayForeground': {
            name: 'Drifting Decay',
            description: 'White crawlers drift and decay',
            className: 'DriftingDecayForeground'
        }
    },
    presets: {
        'nebulaStarsAnimation': {
            name: 'Nebula + Stars',
            background: 'NebulaBackground',
            foreground: 'CrawlingStarsForeground'
        },
        'nebulaSparksAnimation': {
            name: 'Nebula + Sparks',
            background: 'NebulaBackground',
            foreground: 'SparksForeground'
        },
        'nebulaCollisionAnimation': {
            name: 'Nebula + Collision',
            background: 'NebulaBackground',
            foreground: 'CollisionForeground'
        },
        'purpleRainbowAnimation': {
            name: 'Purple + Rainbow',
            background: 'SolidColorBackground',
            foreground: 'RainbowCircleForeground'
        },
        'pulsingCrawlAnimation': {
            name: 'Pulsing + Crawl',
            background: 'PulsingColorBackground',
            foreground: 'EnhancedCrawlForeground'
        },
        'fragmentationAnimation': {
            name: 'Fragmentation',
            background: 'SolidColorBackground',
            foreground: 'FragmentationForeground'
        },
        'driftingDecayAnimation': {
            name: 'Drifting Decay',
            background: 'SolidColorBackground',
            foreground: 'DriftingDecayForeground'
        },
        'rainbowCircleOnlyAnimation': {
            name: 'Rainbow Circle Only',
            background: null,
            foreground: 'RainbowCircleForeground'
        }
    }
};

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    console.log('LED Effects Gallery initialized');
    setupEventListeners();
});

// Set up event listeners for all buttons
function setupEventListeners() {
    // Get all Wokwi buttons
    const wokwiButtons = document.querySelectorAll('.btn-wokwi');
    wokwiButtons.forEach(button => {
        button.addEventListener('click', handleWokwiClick);
    });

    // Get all Upload buttons
    const uploadButtons = document.querySelectorAll('.btn-upload');
    uploadButtons.forEach(button => {
        button.addEventListener('click', handleUploadClick);
    });

    // Add hover effects to cards
    const cards = document.querySelectorAll('.effect-card');
    cards.forEach(card => {
        card.addEventListener('click', (e) => {
            // Don't trigger if clicking a button
            if (e.target.tagName === 'BUTTON') return;

            card.classList.toggle('selected');
        });
    });
}

// Handle Wokwi simulation button click
async function handleWokwiClick(event) {
    const button = event.target;
    const card = button.closest('.effect-card');

    if (card.dataset.combo) {
        // This is a preset combination
        const comboName = card.dataset.combo;
        console.log(`Running Wokwi simulation for: ${comboName}`);
        await runWokwiSimulation(comboName);
    } else {
        // This is an individual effect
        const effectType = card.dataset.effectType;
        const effectName = card.dataset.effectName;
        console.log(`Running Wokwi simulation for ${effectType}: ${effectName}`);
        await runWokwiSimulation(effectName, effectType);
    }
}

// Handle Upload to Arduino button click
async function handleUploadClick(event) {
    const button = event.target;
    const card = button.closest('.effect-card');

    if (card.dataset.combo) {
        // This is a preset combination
        const comboName = card.dataset.combo;
        console.log(`Uploading to Arduino: ${comboName}`);
        await uploadToArduino(comboName);
    } else {
        // This is an individual effect
        const effectType = card.dataset.effectType;
        const effectName = card.dataset.effectName;
        console.log(`Uploading to Arduino ${effectType}: ${effectName}`);
        await uploadToArduino(effectName, effectType);
    }
}

// Run Wokwi simulation
// This will modify effects.ino to select the animation and compile for Wokwi
async function runWokwiSimulation(effectOrCombo, type = null) {
    console.log('runWokwiSimulation called with:', effectOrCombo, type);

    // TODO: Implement Wokwi simulation logic
    // Steps:
    // 1. Modify effects.ino to uncomment the desired animation in loop()
    // 2. Compile the code
    // 3. Open/refresh Wokwi simulation

    alert(`Wokwi simulation for ${effectOrCombo} - Implementation coming soon!`);
}

// Upload to Arduino
// This will modify the code and upload using pio-upload
async function uploadToArduino(effectOrCombo, type = null) {
    console.log('uploadToArduino called with:', effectOrCombo, type);

    // TODO: Implement Arduino upload logic
    // Steps:
    // 1. Modify effects.ino to uncomment the desired animation in loop()
    // 2. Use platformio to compile and upload

    alert(`Upload to Arduino for ${effectOrCombo} - Implementation coming soon!`);
}

// Utility function to show loading state
function showLoading(button) {
    button.disabled = true;
    button.textContent = 'Loading...';
}

// Utility function to hide loading state
function hideLoading(button, originalText) {
    button.disabled = false;
    button.textContent = originalText;
}

// Export functions for potential future use
window.LEDGallery = {
    runWokwiSimulation,
    uploadToArduino,
    effectsData
};
