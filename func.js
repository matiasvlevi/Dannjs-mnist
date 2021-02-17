function findLargest(arr) {
    let record = 0;
    let bestIndex = 0;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] > record) {
            record = arr[i];
            bestIndex = i;
        }
    }

    return bestIndex;
}
function makeEmpty(num) {
    let arr = [];
    for (let i = 0; i < num; i++) {
        arr[i] = 0;
    }
    return arr;
}
module.exports = {
    findLargest,
    makeEmpty
}
