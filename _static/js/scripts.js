
function scroll_to(clicked_link, nav_height) {
	var element_class = clicked_link.attr('href').replace('#', '.');
	var scroll_to = 0;
	if(element_class != '.top-content') {
		element_class += '-container';
		scroll_to = $(element_class).offset().top - nav_height;
	}
	if($(window).scrollTop() != scroll_to) {
		$('html, body').stop().animate({scrollTop: scroll_to}, 1000);
	}
}


jQuery(document).ready(function() {
	
	/*
	    Sidebar
	*/
	$('.dismiss, .overlay').on('click', function() {
        $('.sidebar').removeClass('active');
        $('.overlay').removeClass('active');
    });

    $('.open-menu').on('click', function(e) {
    	e.preventDefault();
        $('.sidebar').addClass('active');
        $('.overlay').addClass('active');
        // close opened sub-menus
        $('.collapse.show').toggleClass('show');
        $('a[aria-expanded=true]').attr('aria-expanded', 'false');
    });
    /* change sidebar style */
	$('a.btn-customized-dark').on('click', function(e) {
		e.preventDefault();
		$('.sidebar').removeClass('light');
	});
	$('a.btn-customized-light').on('click', function(e) {
		e.preventDefault();
		$('.sidebar').addClass('light');
	});
	/* replace the default browser scrollbar in the sidebar, in case the sidebar menu has a height that is bigger than the viewport */
	$('.sidebar').mCustomScrollbar({
		theme: "minimal-dark"
	});
	
	/*
	    Navigation
	*/
	$('a.scroll-link').on('click', function(e) {
		e.preventDefault();
		scroll_to($(this), 0);
	});
	
	$('.to-top a').on('click', function(e) {
		e.preventDefault();
		if($(window).scrollTop() != 0) {
			$('html, body').stop().animate({scrollTop: 0}, 1000);
		}
	});
	/* make the active menu item change color as the page scrolls up and down */
	/* we add 2 waypoints for each direction "up/down" with different offsets, because the "up" direction doesn't work with only one waypoint */
	$('.section-container').waypoint(function(direction) {
		if (direction === 'down') {
			$('.menu-elements li').removeClass('active');
			$('.menu-elements a[href="#' + this.element.id + '"]').parents('li').addClass('active');
			//console.log(this.element.id + ' hit, direction ' + direction);
		}
	},
	{
		offset: '0'
	});
	$('.section-container').waypoint(function(direction) {
		if (direction === 'up') {
			$('.menu-elements li').removeClass('active');
			$('.menu-elements a[href="#' + this.element.id + '"]').parents('li').addClass('active');
			//console.log(this.element.id + ' hit, direction ' + direction);
		}
	},
	{
		offset: '-5'
	});


    
    /*
	    Wow
	*/
	new WOW().init();
	
	/*
	    Contact form
	*/
	$('.section-6-form form input[type="text"], .section-6-form form textarea').on('focus', function() {
		$('.section-6-form form input[type="text"], .section-6-form form textarea').removeClass('input-error');
	});
	$('.section-6-form form').submit(function(e) {
		e.preventDefault();
	    $('.section-6-form form input[type="text"], .section-6-form form textarea').removeClass('input-error');
	    var postdata = $('.section-6-form form').serialize();
	    $.ajax({
	        type: 'POST',
	        url: 'assets/contact.php',
	        data: postdata,
	        dataType: 'json',
	        success: function(json) {
	            if(json.emailMessage != '') {
	                $('.section-6-form form .contact-email').addClass('input-error');
	            }
	            if(json.subjectMessage != '') {
	                $('.section-6-form form .contact-subject').addClass('input-error');
	            }
	            if(json.messageMessage != '') {
	                $('.section-6-form form textarea').addClass('input-error');
	            }
	            if(json.emailMessage == '' && json.subjectMessage == '' && json.messageMessage == '') {
	                $('.section-6-form form').fadeOut('fast', function() {
	                    $('.section-6-form').append('<p>Thanks for contacting us! We will get back to you very soon.</p>');
	                    $('.section-6-container').backstretch("resize");
	                });
	            }
	        }
	    });
	});
	
});
