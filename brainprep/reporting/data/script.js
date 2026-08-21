
/**
 * Menu handling.
 */
$(function() {

  var siteSticky = function() {
		$(".js-sticky-header").sticky({topSpacing:0});
	};
	siteSticky();

	var siteMenuClone = function() {

		$('.js-clone-nav').each(function() {
			var $this = $(this);
			$this.clone().attr('class', 'site-nav-wrap').appendTo('.site-mobile-menu-body');
		});


		setTimeout(function() {
			
			var counter = 0;
      $('.site-mobile-menu .has-children').each(function(){
        var $this = $(this);
        
        $this.prepend('<span class="arrow-collapse collapsed">');

        $this.find('.arrow-collapse').attr({
          'data-toggle' : 'collapse',
          'data-target' : '#collapseItem' + counter,
        });

        $this.find('> ul').attr({
          'class' : 'collapse',
          'id' : 'collapseItem' + counter,
        });

        counter++;

      });

    }, 1000);

		$('body').on('click', '.arrow-collapse', function(e) {
      var $this = $(this);
      if ( $this.closest('li').find('.collapse').hasClass('show') ) {
        $this.removeClass('active');
      } else {
        $this.addClass('active');
      }
      e.preventDefault();  
      
    });

		$(window).resize(function() {
			var $this = $(this),
				w = $this.width();

			if ( w > 768 ) {
				if ( $('body').hasClass('offcanvas-menu') ) {
					$('body').removeClass('offcanvas-menu');
				}
			}
		})

		$('body').on('click', '.js-menu-toggle', function(e) {
			var $this = $(this);
			e.preventDefault();

			if ( $('body').hasClass('offcanvas-menu') ) {
				$('body').removeClass('offcanvas-menu');
				$this.removeClass('active');
			} else {
				$('body').addClass('offcanvas-menu');
				$this.addClass('active');
			}
		}) 

		// click outside offcanvas
		$(document).mouseup(function(e) {
	    var container = $(".site-mobile-menu");
	    if (!container.is(e.target) && container.has(e.target).length === 0) {
	      if ( $('body').hasClass('offcanvas-menu') ) {
					$('body').removeClass('offcanvas-menu');
				}
	    }
		});
	}; 
	siteMenuClone();

});



/**
 * A class representing a carousel that cycles through different objects.
 */
class Carousel {
    /**
     * Creates a new Carousel instance.
     * @param {string} uid - A unique identifier for the masker report.
     * @param {number[]} displayed_objects - An array of IDs to cycle through.
     */
    constructor(uid, displayed_objects) {
        /** @private {string} */
        this.uid = uid;
        console.log("uid=" + this.uid);

        /** @private {number[]} */
        this.displayed_objects = displayed_objects;
        console.log("names=" + this.displayed_objects);

        /** @private {number} */
        this.current_obj_idx = 0;
        console.log("new index=" + this.current_obj_idx);

        /** @private {number} */
        this.number_objs = displayed_objects.length;
        console.log("size=" + this.number_objs);

        this.init();
    }

    /**
     * Initializes the carousel by setting up event listeners and displaying the first object.
     */
    init() {
        this.showObj(this.current_obj_idx);

        let prevButton = document.querySelector(`#prev-btn-${this.uid}`);
        let nextButton = document.querySelector(`#next-btn-${this.uid}`);

        if (prevButton) prevButton.addEventListener("click", () => this.displayPrevious());
        if (nextButton) nextButton.addEventListener("click", () => this.displayNext());

        this.bindKeyboardEvents();
    }

    /**
     * Displays the object at the given index and hides all others.
     * @param {number} index - The index of the map to display.
     *
     * for sphere masker report this adapts the full title in the carousel
     */
    showObj(index) {
        this.displayed_objects.forEach((_, i) => {
            let mapElement = document.getElementById(`carousel-obj-${this.uid}-${i}`);
            if (mapElement) {
                mapElement.style.display = i === index ? "block" : "none";
            }
        });

        let compElement = document.getElementById(`comp-${this.uid}`);
        if (compElement) {
            compElement.innerHTML = this.displayed_objects[index];
        }
    }

    /**
     * Advances the carousel to the next object.
     *
     * using % modulo to ensure we 'wrap' back to start in the carousel
    */
    displayNext() {
        this.current_obj_idx = (this.current_obj_idx + 1) % this.number_objs;
        console.log("new index=" + this.current_obj_idx);
        this.showObj(this.current_obj_idx);
    }

    /**
     * Moves the carousel to the previous object.
     *
     * using % modulo to ensure we 'wrap' back to start in the carousel
    */
    displayPrevious() {
        this.current_obj_idx = (this.current_obj_idx - 1 + this.number_objs) % this.number_objs;
        console.log("new index=" + this.current_obj_idx);
        this.showObj(this.current_obj_idx);
    }

    /**
     * Binds carousel to right and left arrow keys to cycle through carousel.
    */
    bindKeyboardEvents() {
        document.addEventListener("keydown", (event) => {
            if (event.key === "ArrowRight") {
                this.displayNext();
            } else if (event.key === "ArrowLeft") {
                this.displayPrevious();
            }
        });
    }
}



/**
 * Initializes a reusable and interactive Canvas scatter plot.
 *
 * @param {Object} options - Chart configuration options.
 * @param {string} options.canvasId - The ID of the <canvas> element.
 * @param {Array} options.data - Array of data points {x, y, img}.
 * @param {Object} [options.limits=null] - Axis boundaries {minX, maxX, minY, maxY}.
 * @param {Object} [options.domTargets] - HTML element IDs for click actions {imgId, placeholderId}.
 * @param {string} [options.xLabel=""] - Title text for the X axis.
 * @param {string} [options.yLabel=""] - Title text for the Y axis.
 * @param {number} [options.radius=8] - Dots radius in pixels.
 * @param {number} [options.padding=50] - Inner chart padding in pixels.
 */
function createInteractiveChart({
    canvasId,
    data,
    limits = null,
    domTargets = { imgId: 'clicked-image', placeholderId: 'placeholder' },
    xLabel = "",
    yLabel = "",
    radius = 8,
    padding = 50
}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return console.error(`Canvas with ID "${canvasId}" not found.`);

    const ctx = canvas.getContext('2d');

    // Clean Auto-scaling with Epsilon protection
    let minX, maxX, minY, maxY;
    if (limits) {
        ({ minX, maxX, minY, maxY } = limits);
    } else {
        const xValues = data.map(p => p.x);
        const yValues = data.map(p => p.y);
        minX = Math.min(...xValues);
        maxX = Math.max(...xValues);
        minY = Math.min(...yValues);
        maxY = Math.max(...yValues);

        const EPSILON = 0.001;
        if (Math.abs(maxX - minX) < EPSILON) {
            const padX = maxX - minX === 0 ? 1 : Math.abs(maxX - minX) * 2;
            console.log("padX=" + padX);
            minX -= padX; maxX += padX;
        }
        if (Math.abs(maxY - minY) < EPSILON) {
            const padY = maxY - minY === 0 ? 1 : Math.abs(maxY - minY) * 2;
            console.log("padY=" + padY);
            minY -= padY; maxY += padY;
        }
        console.log("minX=" + minX+ ", maxX=" + maxX);
        console.log("minY=" + minY+ ", maxY=" + maxY);
    }

    // Pre-calculate and store canvas coordinates for each data point
    const points = data.map(p => ({
        ...p,
        canvasX: padding + ((p.x - minX) / (maxX - minX)) * (canvas.width - 2 * padding),
        canvasY: canvas.height - padding - ((p.y - minY) / (maxY - minY)) * (canvas.height - 2 * padding)
    }));

    function draw() {
        // Clear canvas for fresh render
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 1. Gridlines and Text Styles
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';

        // X-Axis Grid (Vertical lines)
        ctx.textAlign = 'center';
        for (let x = minX; x <= maxX; x++) {
            const cx = padding + ((x - minX) / (maxX - minX)) * (canvas.width - 2 * padding);
            ctx.beginPath();
            ctx.moveTo(cx, padding);
            ctx.lineTo(cx, canvas.height - padding);
            ctx.stroke();
            // ctx.fillText(x, cx, canvas.height - padding + 20);
        }

        // Y-Axis Grid (Horizontal lines)
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let y = minY; y <= maxY; y += 10) {
            const cy = canvas.height - padding - ((y - minY) / (maxY - minY)) * (canvas.height - 2 * padding);
            ctx.beginPath();
            ctx.moveTo(padding, cy);
            ctx.lineTo(canvas.width - padding, cy);
            ctx.stroke();
            // ctx.fillText(y, padding - 10, cy);
        }

        // 2. Main Axes Lines
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, canvas.height - padding);
        ctx.lineTo(canvas.width - padding, canvas.height - padding);
        ctx.stroke();

        // 3. Draw Axis Legends
        ctx.fillStyle = '#111';
        ctx.font = 'bold 12px Arial';

        if (xLabel) {
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText(xLabel, padding + (canvas.width - 2 * padding) / 2, canvas.height - padding + 26);
        }

        if (yLabel) {
            ctx.save();
            ctx.translate(padding - 35, padding + (canvas.height - 2 * padding) / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.fillText(yLabel, 0, 0);
            ctx.restore();
        }

        // 4. Render Data Points
        points.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.canvasX, p.canvasY, radius, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(33, 150, 243, 0.8)';
            ctx.fill();
            ctx.strokeStyle = '#0d47a1';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });
    }

    // Native Click Event Listener
    canvas.addEventListener('click', function(event) {
        // Get precise mouse cursor position relative to the canvas bounding box
        const rect = canvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;

        // Detect if the mouse cursor coordinates collide with any data point
        const foundPoint = points.find(p => {
            const distance = Math.sqrt((mouseX - p.canvasX) ** 2 + (mouseY - p.canvasY) ** 2);
            return distance <= radius + 4; // Includes a 4-pixel tolerance margin
        });

        // Update DOM elements if a point is clicked successfully
        const popup = document.getElementById(domTargets.popupId);
        if (foundPoint && domTargets) {
            const imgElement = document.getElementById(domTargets.imgId);
            const placeholder = document.getElementById(domTargets.placeholderId);

            if (imgElement) {
                imgElement.src = foundPoint.img;
                imgElement.style.display = 'inline-block';
            }
            if (placeholder) {
                placeholder.style.display = 'none';
            }
            if (popup) {
                popup.innerHTML = `<b>Subject:</b> ${foundPoint.sub}<br><b>Session:</b> ${foundPoint.ses}<br><b>Run:</b> ${foundPoint.run}`;
                popup.style.display = 'block';
                popup.style.left = (event.clientX + 10 - getDetailsLeftPosition()) + 'px';
                popup.style.top = (event.clientY + 10 + window.scrollY - getDetailsTopPosition()) + 'px';
            }
        } else {
            // Hide popup if no point is found
            if (popup) {
                popup.style.display = 'none';
            }
        }
    });

    // Function to get the left position of the details element
    function getDetailsLeftPosition() {
        const detailsElement = canvas.closest('.custom-details');
        if (detailsElement) {
            const rect = detailsElement.getBoundingClientRect();
            return rect.left + window.scrollX;
        }
        return 0;
    }

    // Function to get the top position of the details element
    function getDetailsTopPosition() {
        const detailsElement = canvas.closest('.custom-details');
        if (detailsElement) {
            const rect = detailsElement.getBoundingClientRect();
            return rect.top + window.scrollY;
        }
        return 0;
    }

    // Hide popup when clicking elsewhere
    document.addEventListener('click', function(event) {
        const popup = document.getElementById(domTargets.popupId);
        if (popup && !canvas.contains(event.target)) {
            popup.style.display = 'none';
        }
    });

    // Initial call to render the chart
    draw();
}


/**
 * Scrolls to a specified section on the page, accounting for the height of a fixed navbar.
 * This function smoothly scrolls the page to the top of the specified section, adjusting
 * the scroll position to ensure the section is visible below the fixed navbar.
 *
 * @param {string} sectionId - The ID of the section to scroll to.
 *
 * @example
 * // Scroll to the section with the ID 'mySection'
 * scrollToSection('mySection');
 */
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        // Calculate the position to scroll to, accounting for the navbar height
        const navbarHeight = document.querySelector(".site-navbar").offsetHeight;
        const sectionPosition = section.getBoundingClientRect().top + window.scrollY - navbarHeight;

        // Scroll to the calculated position
        window.scrollTo({
            top: sectionPosition,
            behavior: "smooth"
        });
    }
}
