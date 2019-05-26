<?php

$img = "/srv/cv_core/uploads/";
if (!empty($_POST["params"])) {
    echo '<li>';
    var_dump($_POST);
    echo '</li>';
    
    $img.=$_POST["params"][0];

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
    if (file_exists($img.".jpeg")) {
        $o = file_get_contents($img.".jpeg");
        if (!empty($o)) {
            echo "<li><div id='prev'><img id='img2' title='Megtalált él kiemelve' src='data:image/jpeg;base64, ";
            echo base64_encode($o);
            echo "' /></div></li>";
        }
    }
    //vágott warped image
    if (file_exists($img.".jpg")) {
        $o = file_get_contents($img.".jpg");
        if (!empty($o)) {
            echo "<li><div id='prev'><img id='img2' title='Megtalált él megvágva' src='data:image/jpeg;base64, ";
            echo base64_encode($o);
            echo "' /></div></li>";
        }
    }
    echo '<li>';
    echo 'DONE';
    echo '</li>';
}
exit();
?>


