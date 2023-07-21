
function fabric_selected_gallery_index(elem_id) {
    const el = document.getElementById(elem_id);
    const buttons = el.querySelectorAll('.thumbnail-item.thumbnail-small');
    const button = el.querySelector('.thumbnail-item.thumbnail-small.selected');

    let result = -1;
    buttons.forEach((v, i) => {
        if (v == button) {
            result = i;
        }
    });

    console.log(result);
    console.log(button);
    console.log(buttons);

    return result;
}
