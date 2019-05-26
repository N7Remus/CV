<html>
    <head>
        <meta charset="UTF-8">
        <title>Boredom</title>
        <script src="jquery.js" type="text/javascript"></script>
        <script src="slider.js" type="text/javascript"></script>    
        <link href="css_reset.css" type="text/css" rel="stylesheet" />
        <link href="slider.css" type="text/css" rel="stylesheet" />
    </head>
    <body>
        <script>
            $(document).ready(function () {
                $("#btn_r").click(function () {
                    var name = $("#img1").attr('name')
                    var params = [name];
                    if ($('#median').is(':checked')) {
                        params.push("median");
                    }
                    if ($('#gauss').is(':checked')) {
                        params.push("gauss");
                    }
                    if ($('#bilinear').is(':checked')) {
                        params.push("bilinear");
                    }
                    if ($('#invert').is(':checked')) {
                        params.push("invert");
                    }
                    if ($('#clahe').is(':checked')) {
                        params.push("clahe");
                    }
                    if ($('#kvantal').is(':checked')) {
                        params.push("kvantal");
                    }
                    console.log(params);
                    $.post("api.php", {params: params}).done(function (data) {
                        alert("Data Loaded: " + data);
                        $("#output").html(data);
                    });
                })
            });


        </script>
        <style>

            * {
                box-sizing:border-box;
            }

            body {
                font-family:'Arial';
            }

            header {
                background-color:#000000;
                padding:15px;
                position:fixed;
                top:0;
                left:0;
                z-index:9;
                width:100%;
            }

            header * {
                text-align:center;
                color:#ffffff;
            }

            header input[type=submit] {
                border:solid 1px #ffffff;
                padding:10px;
                cursor:pointer;
                transition:0.3s;
            }

            header input[type=submit]:hover {
                background-color:#ffffff;
                color:#000000;
            }

            .container {
                width:1274px;
                margin:0 auto;
                padding:104px 0;
                display:flex;
                flex-wrap:wrap;
                justify-content:space-between;
                padding-top: 72px;
            }

            .panel {
                width:50%;
                float:left;
                background-color:#f0f0f0;

                padding:32px;


            }

            .panel:first-child {
                border:solid 1px #f0f0f0;
                background:none;
                text-align:center;
                width:calc(50% - 20px);
                display:flex;
                justify-content:center;
                align-items:center;
            }


            div.file {
                font-size:48px;
                color:#a0a0a0;
                border-bottom:solid 3px #a0a0a0;
                margin-bottom:10px;
                padding-bottom:5px;
            }

            img {
                max-width:100%;
                width:auto;
                height:auto;
                box-shadow:0 10px 10px rgba(0,0,0,0.2);
            }

            footer {
                padding:15px;
                position:fixed;
                color:#ffffff;
                background-color:#a0a0a0;
                bottom:0;
                left:0;
                width:100%;
                z-index:9;
                text-align:center;
            }

            .table-wrap {
                padding:20px;
                background-color:#ffffff;
                box-shadow:0 10px 10px rgba(0,0,0,0.2);
                margin:32px 0;
                width:100%;
            }

            table {
                border-spacing: 1px;
                border-collapse:separate;
                width:100%;
            }

            table tr td {
                padding:2px 0;
            }

            table tr td:first-child {
                font-weight:bold;
            }
            button {
                background-color: #4CAF50; /* Green */
                border: none;
                color: white;
                padding: 16px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                -webkit-transition-duration: 0.4s; /* Safari */
                transition-duration: 0.4s;
                cursor: pointer;
            }
            .button2 {
                width: 100%;
                background-color: white; 
                color: black; 
                border: 2px solid #008CBA;
            }

            .button2:hover {
                background-color: #008CBA;
                color: white;
            }



        </style>

        <header>
            <form action="" method="post" enctype="multipart/form-data">
                Válassz képet a feltöltéshez:
                <input type="file" name="fileToUpload" id="fileToUpload" >
                <input type="submit" value="Kép Feltöltése" name="submit">
            </form>
        </header>
        <?php
        $date = date("Y_m_d_His");

        $target_dir = "/srv/cv_tmp/";
        //$_FILES["fileToUpload"]["name"] as $key => $value)
        $target_file = $target_dir . basename($_FILES["fileToUpload"]["name"]);
        $imageFileType = strtolower(pathinfo($target_file, PATHINFO_EXTENSION));
        $check = getimagesize($_FILES["fileToUpload"]["tmp_name"]);
        if ($check == false) {
            $msg = "A " . $_FILES["fileToUpload"]["name"] . " file nem kép";
            $error = TRUE;
        }
        ?>
        <div class="container">
            <div class="panel">
                <?php
                if ($error) {
                    echo '<img id="img1" title="Eredeti kép" src="img1.jpg">';
                } else {
                    //original img
                    $img_path = $_FILES["fileToUpload"]["tmp_name"];
                    $o = file_get_contents($img_path);
                    if (!empty($o)) {
                        //$_FILES["fileToUpload"]["tmp_name"][$key], time()+3600);/* expire in 1 hour */
                        echo "<div id='prev'><img id='img1' name=$date src='data:image/jpeg;base64, ";
                        echo base64_encode($o);
                        echo "' /></div>";
                    }
                }
                ?>


            </div>
            <div class="panel">
                <?php
                if (!empty($_FILES["fileToUpload"]["name"])) {
                    echo '<div class="file">' . $_FILES["fileToUpload"]["name"] . '</div>';
                } else {
                    echo '<div class="file">Tölts fel képet!</div>';
                }
                ?>

                <?php
                if (empty($_FILES["fileToUpload"]["name"])) {
                    
                } else if ($error) {
                    echo $msg;
                } else {
                    $check = getimagesize($_FILES["fileToUpload"]["tmp_name"]);
                    if ($check !== false) {
                        echo "<div>A " . $_FILES["fileToUpload"]["name"] . " kép mime típusa: - " . $check["mime"] . ".</div>";

                        // image-feldolgozás
                        $new_img = "/srv/cv_core/uploads/" . $date;
                        copy($_FILES["fileToUpload"]["tmp_name"], $new_img);
                        echo "<div id='scan_op' >";
                        echo "</div>";
                    } else {
                        $error = TRUE;
                    }
                    ?>
                    <div class="table-wrap">
                        <table>
                            <tr>
                                <td>Median filterezés - Só és bors zaj-ellen:</td>
                                <td><input id="median" type="checkbox"></td>
                            </tr>
                            <tr>
                                <td>Gaussian zajcsökkentés - Elmossa a képet</td>
                                <td><input id="gauss" type="checkbox"></td>
                            </tr>
                            <tr>
                                <td>Bilineáris filter - textúrákat csökkenti, az éleket meghagyja</td>
                                <td><input id="bilinear" type="checkbox"></td>
                            </tr>
                            <tr>
                                <td>Kép invertálás - negatív képet készít</td>
                                <td><input id="invert" type="checkbox"></td>
                            </tr>
                            <tr>
                                <td>CLAHE (Contrast Limited Adaptive Histogram Equalization)</td>
                                <td><input id="clahe" type="checkbox"></td>
                            </tr>
                            <tr>
                                <td>Kvantálás-csökkenti a képen látható különböző színek számát.</td>
                                <td><input id="kvantal" type="checkbox"></td>
                            </tr>
                            <tr>
                                <td>
                                    <button type="button" id="btn_r" class="button button2" >Frissítés</button>
                                </td>
                            </tr>
                        </table>
                    </div>

                    <div class="table-wrap">
                        <table>
                            <?php
                            echo "<tr><td>";
                            echo str_replace(";", "</td></tr><tr><td>", shell_exec("export PYTHONIOENCODING=UTF-8;python3 /srv/cv_core/virtualenvironment/scan.py -i " . "'$img_path' -o '$new_img' 2>&1"));
                            echo "</td></tr>"
                            ?>
    <!--                            <tr class="done">
                                <td>DONE!</td>
                                <td>&nbsp;</td>
                            </tr>-->
                        </table>
                    </div>

                    <?php
                    // KIMENET  
                    if (file_exists($new_img)) {
                        ?>

                        <div id="slider">
                            <a href="javascript:void(0);" class="control_next">></a>
                            <a href="javascript:void(0);" class="control_prev"><</a>
                            <ul id="output">

                                <?php
                                $img = $new_img;

                                //original image
                                if (file_exists($img)) {
                                    $o = file_get_contents($img);
                                    if (!empty($o)) {
                                        echo "<li><div id='prev'><img id='img2' title='Eredeti kép' src='data:image/jpeg;base64, ";
                                        echo base64_encode($o);
                                        echo "' /></div></li>";
                                    }
                                }

                                //kimelt warped image
                                if (file_exists($img . ".jpeg")) {
                                    $o = file_get_contents($img . ".jpeg");
                                    if (!empty($o)) {
                                        echo "<li><div id='prev'><img id='img2' title='Megtalált él kiemelve' src='data:image/jpeg;base64, ";
                                        echo base64_encode($o);
                                        echo "' /></div></li>";
                                    }
                                }
                                //vágott warped image
                                if (file_exists($img . ".jpg")) {
                                    $o = file_get_contents($img . ".jpg");
                                    if (!empty($o)) {
                                        echo "<li><div id='prev'><img id='img2' title='Megtalált él megvágva' src='data:image/jpeg;base64, ";
                                        echo base64_encode($o);
                                        echo "' /></div></li>";
                                    }
                                }
                                echo '<li>';
                                echo 'DONE';
                                echo '</li>';
                                ?>

                                <!-- 
                                <li style="background: #aaa;">SLIDE 2</li>
                                                           <li>SLIDE 3</li>
                                                            <li style="background: #aaa;">SLIDE 4</li>-->
                            </ul>  
                        </div>
                        <?php
                    }
                    ?>

                    <?php
                }
                ?>


            </div>
        </div>


        <footer>Rémai Gábor - Gépi Látás <a href="https://github.com/N7Remus/CV" ><img style="width: 20px" src="GitHub-Mark-64px.png"></a>&nbsp;</footer>
    </body>
</html>
