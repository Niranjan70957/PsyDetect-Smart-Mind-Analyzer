/* ============================================================
   PsyDetect Smart Mind Analyzer
   Interactive JavaScript
   ============================================================ */

document.addEventListener("DOMContentLoaded", function () {

    /* ========================================================
       Navbar Scroll Effect
       ======================================================== */

    const navbar = document.querySelector(".main-navbar");

    window.addEventListener("scroll", function () {

        if (!navbar) return;

        if (window.scrollY > 50) {
            navbar.style.padding = "8px 0";
        } else {
            navbar.style.padding = "14px 0";
        }

    });


    /* ========================================================
       Smooth Scroll
       ======================================================== */

    document.querySelectorAll('a[href^="#"]').forEach(
        function (anchor) {

            anchor.addEventListener("click", function (event) {

                const target = document.querySelector(
                    this.getAttribute("href")
                );

                if (target) {

                    event.preventDefault();

                    target.scrollIntoView({
                        behavior: "smooth"
                    });

                }

            });

        }
    );


    /* ========================================================
       Audio Upload / Drag and Drop
       ======================================================== */

    const dropZone = document.getElementById("dropZone");
    const audioFile = document.getElementById("audioFile");
    const fileName = document.getElementById("fileName");

    if (dropZone && audioFile) {

        dropZone.addEventListener(
            "click",
            function () {
                audioFile.click();
            }
        );


        audioFile.addEventListener(
            "change",
            function () {

                if (this.files.length > 0) {

                    showFileName(
                        this.files[0]
                    );

                }

            }
        );


        dropZone.addEventListener(
            "dragover",
            function (event) {

                event.preventDefault();

                dropZone.classList.add(
                    "dragover"
                );

            }
        );


        dropZone.addEventListener(
            "dragleave",
            function () {

                dropZone.classList.remove(
                    "dragover"
                );

            }
        );


        dropZone.addEventListener(
            "drop",
            function (event) {

                event.preventDefault();

                dropZone.classList.remove(
                    "dragover"
                );

                const files =
                    event.dataTransfer.files;

                if (files.length > 0) {

                    audioFile.files = files;

                    showFileName(
                        files[0]
                    );

                }

            }
        );

    }


    function showFileName(file) {

        if (!fileName) return;

        const allowed = [
            "audio/wav",
            "audio/x-wav",
            "audio/mpeg",
            "audio/ogg",
            "audio/flac"
        ];

        fileName.innerHTML =
            '<i class="bi bi-file-earmark-music me-2"></i>' +
            file.name;

    }


    /* ========================================================
       Button Loading Effect
       ======================================================== */

    document.querySelectorAll("form").forEach(
        function (form) {

            form.addEventListener(
                "submit",
                function () {

                    const button =
                        form.querySelector(
                            'button[type="submit"]'
                        );

                    if (
                        button &&
                        form.querySelector(
                            'input[type="file"]'
                        )
                    ) {

                        button.disabled = true;

                        button.innerHTML =
                            '<span class="spinner-border spinner-border-sm me-2"></span>' +
                            'Analyzing...';

                    }

                }
            );

        }
    );


    /* ========================================================
       Auto Hide Flash Messages
       ======================================================== */

    setTimeout(
        function () {

            document.querySelectorAll(
                ".alert"
            ).forEach(
                function (alert) {

                    if (
                        typeof bootstrap !== "undefined"
                    ) {

                        const instance =
                            bootstrap.Alert.getOrCreateInstance(
                                alert
                            );

                        instance.close();

                    }

                }
            );

        },
        5000
    );


    /* ========================================================
       Simple Reveal Animation
       ======================================================== */

    const revealElements =
        document.querySelectorAll(
            ".feature-card, .workflow-card, .dashboard-card"
        );

    const observer =
        new IntersectionObserver(
            function (entries) {

                entries.forEach(
                    function (entry) {

                        if (entry.isIntersecting) {

                            entry.target.style.opacity = "1";
                            entry.target.style.transform =
                                "translateY(0)";

                        }

                    }
                );

            },
            {
                threshold: 0.1
            }
        );


    revealElements.forEach(
        function (element) {

            element.style.opacity = "0";
            element.style.transform =
                "translateY(20px)";
            element.style.transition =
                "all .6s ease";

            observer.observe(element);

        }
    );

});