const video = document.getElementById('webcam');

const overlay = document.getElementById('overlay');

const uploadedImage = document.getElementById('uploadedImage');

const uploadedVideo = document.getElementById('uploadedVideo');

const imageUpload = document.getElementById('imageUpload');

const videoUpload = document.getElementById('videoUpload');

const imageBtn = document.getElementById('imageBtn');

const videoBtn = document.getElementById('videoBtn');

const liveBtn = document.getElementById('liveBtn');

const cameraWrapper = document.querySelector('.camera-wrapper');

const knownBox = document.querySelector('.known');

const unknownBox = document.querySelector('.unknown');

const criminalBox = document.querySelector('.criminal');

const hiddenCanvas = document.createElement('canvas');

const hiddenCtx = hiddenCanvas.getContext('2d');

const ctx = overlay.getContext('2d');

let processing = false;

let faceDetected = false;

let currentState = 'idle';

let lastFaceLocation = null;

let liveStream = null;

let currentSource = video;

function resizeOverlay(){

    overlay.width = currentSource.clientWidth;

    overlay.height = currentSource.clientHeight;

    if(currentSource.videoWidth){

        hiddenCanvas.width = currentSource.videoWidth;

        hiddenCanvas.height = currentSource.videoHeight;
    }

    if(currentSource.naturalWidth){

        hiddenCanvas.width = currentSource.naturalWidth;

        hiddenCanvas.height = currentSource.naturalHeight;
    }
}

window.addEventListener('resize',resizeOverlay);

function hideBoxes(){

    knownBox.style.display = 'none';

    unknownBox.style.display = 'none';

    criminalBox.style.display = 'none';
}

function clearOverlay(){

    ctx.clearRect(0,0,overlay.width,overlay.height);
}

function setIdleState(){

    hideBoxes();

    clearOverlay();

    cameraWrapper.classList.remove('scanning-active');

    faceDetected = false;

    currentState = 'idle';

    lastFaceLocation = null;
}

function drawFaceFrame(face,state='scan'){

    if(!face){

        clearOverlay();

        return;
    }

    const scaleX = overlay.width / hiddenCanvas.width;

    const scaleY = overlay.height / hiddenCanvas.height;

    const x = face.left * scaleX;

    const y = face.top * scaleY;

    const w = (face.right - face.left) * scaleX;

    const h = (face.bottom - face.top) * scaleY;

    const pad = 25;

    const bx = x - pad;

    const by = y - pad;

    const bw = w + pad * 2;

    const bh = h + pad * 2;

    clearOverlay();

    let color = '#00ff88';

    if(state === 'unknown'){

        color = '#ffd000';
    }

    if(state === 'criminal'){

        color = '#ff2f2f';
    }

    ctx.save();

    ctx.shadowColor = color;

    ctx.shadowBlur = 18;

    ctx.strokeStyle = color;

    ctx.lineWidth = 4;

    const corner = 40;

    ctx.beginPath();

    ctx.moveTo(bx,by+corner);
    ctx.lineTo(bx,by);
    ctx.lineTo(bx+corner,by);

    ctx.moveTo(bx+bw-corner,by);
    ctx.lineTo(bx+bw,by);
    ctx.lineTo(bx+bw,by+corner);

    ctx.moveTo(bx,by+bh-corner);
    ctx.lineTo(bx,by+bh);
    ctx.lineTo(bx+corner,by+bh);

    ctx.moveTo(bx+bw-corner,by+bh);
    ctx.lineTo(bx+bw,by+bh);
    ctx.lineTo(bx+bw,by+bh-corner);

    ctx.stroke();

    ctx.beginPath();

    ctx.moveTo(bx-35,by+bh/2);
    ctx.lineTo(bx+20,by+bh/2);

    ctx.moveTo(bx+bw-20,by+bh/2);
    ctx.lineTo(bx+bw+35,by+bh/2);

    ctx.stroke();

    const lineY = by + ((Date.now()/4) % bh);

    const gradient = ctx.createLinearGradient(
        bx,
        lineY,
        bx+bw,
        lineY
    );

    gradient.addColorStop(0,'transparent');

    gradient.addColorStop(0.5,color);

    gradient.addColorStop(1,'transparent');

    ctx.beginPath();

    ctx.strokeStyle = gradient;

    ctx.lineWidth = 3;

    ctx.moveTo(bx,lineY);

    ctx.lineTo(bx+bw,lineY);

    ctx.stroke();

    ctx.beginPath();

    ctx.arc(
        bx + bw/2,
        by + bh/2,
        7,
        0,
        Math.PI*2
    );

    ctx.fillStyle = color;

    ctx.fill();

    ctx.lineWidth = 1;

    ctx.strokeStyle = color + '55';

    for(let i=1;i<4;i++){

        ctx.beginPath();

        ctx.moveTo(
            bx + (bw/4)*i,
            by
        );

        ctx.lineTo(
            bx + (bw/4)*i,
            by+bh
        );

        ctx.stroke();

        ctx.beginPath();

        ctx.moveTo(
            bx,
            by + (bh/4)*i
        );

        ctx.lineTo(
            bx+bw,
            by + (bh/4)*i
        );

        ctx.stroke();
    }

    ctx.restore();
}

function updateStatus(type,data={}){

    hideBoxes();

    cameraWrapper.classList.add('scanning-active');

    if(type === 'known'){

        knownBox.style.display = 'block';

        knownBox.querySelector('.label').innerHTML =
        `
        ✓ ACCESS GRANTED <br>
        KNOWN PERSON DETECTED <br>
        Name: ${data.name || 'Unknown'} <br>
        Age: ${data.age || 'N/A'}
        `;
    }

    if(type === 'unknown'){

        unknownBox.style.display = 'block';

        unknownBox.querySelector('.label').innerHTML =
        `
        ⚠ UNKNOWN PERSON DETECTED
        `;
    }

    if(type === 'criminal'){

        criminalBox.style.display = 'block';

        criminalBox.querySelector('.label').innerHTML =
        `
        ⛔ CRIMINAL DETECTED <br>
        ALERT ACTIVATED
        `;
    }

    currentState = type;
}

async function getFrameBlob(){

    if(currentSource){

        if(currentSource.tagName === 'IMG'){

            hiddenCanvas.width = currentSource.naturalWidth;

            hiddenCanvas.height = currentSource.naturalHeight;
        }

        else{

            if(!currentSource.videoWidth){

                return null;
            }

            hiddenCanvas.width = currentSource.videoWidth;

            hiddenCanvas.height = currentSource.videoHeight;
        }

        hiddenCtx.drawImage(
            currentSource,
            0,
            0,
            hiddenCanvas.width,
            hiddenCanvas.height
        );

        return new Promise((resolve)=>{

            hiddenCanvas.toBlob((blob)=>{

                resolve(blob);

            },'image/jpeg',0.95);
        });
    }

    return null;
}

async function scanCycle(){

    if(processing){

        return;
    }

    processing = true;

    const blob = await getFrameBlob();

    if(!blob){

        processing = false;

        return;
    }

    try{

        const form = new FormData();

        form.append('frame',blob,'frame.jpg');

        const response = await fetch('/api/scan',{

            method:'POST',

            body:form,

            credentials:'same-origin'
        });

        const result = await response.json();

        processing = false;

        if(!result || result.state === 'idle'){

            setIdleState();

            return;
        }

        faceDetected = true;

        lastFaceLocation = result.face;

        drawFaceFrame(result.face,result.state);

        if(result.state === 'known'){

            updateStatus('known',result);
        }

        else if(result.state === 'unknown'){

            updateStatus('unknown');
        }

        else if(result.state === 'criminal'){

            updateStatus('criminal');
        }
    }

    catch(error){

        console.error(error);

        processing = false;
    }
}

function updateClock(){

    const clock = document.getElementById('clock');

    const now = new Date();

    clock.innerText = now.toLocaleTimeString();
}

function stopCamera(){

    if(liveStream){

        liveStream.getTracks().forEach(track=>{

            track.stop();

        });

        liveStream = null;
    }

    video.srcObject = null;
}

async function startCamera(){

    stopCamera();

    try{

        liveStream = await navigator.mediaDevices.getUserMedia({

            video:{
                width:{ideal:1280},
                height:{ideal:720},
                facingMode:'user'
            },

            audio:false
        });

        video.style.display = 'block';

        uploadedImage.style.display = 'none';

        uploadedVideo.style.display = 'none';

        video.srcObject = liveStream;

        currentSource = video;

        await video.play();

        resizeOverlay();
    }

    catch(error){

        console.error('Camera Error:',error);

        setIdleState();
    }
}

imageBtn.addEventListener('click',()=>{

    imageUpload.click();
});

videoBtn.addEventListener('click',()=>{

    videoUpload.click();
});

liveBtn.addEventListener('click',()=>{

    startCamera();
});

imageUpload.addEventListener('change',(e)=>{

    const file = e.target.files[0];

    if(!file){

        return;
    }

    stopCamera();

    const url = URL.createObjectURL(file);

    uploadedImage.src = url;

    uploadedImage.style.display = 'block';

    uploadedVideo.style.display = 'none';

    video.style.display = 'none';

    currentSource = uploadedImage;

    uploadedImage.onload = ()=>{

        resizeOverlay();

        scanCycle();
    };
});

videoUpload.addEventListener('change',(e)=>{

    const file = e.target.files[0];

    if(!file){

        return;
    }

    stopCamera();

    const url = URL.createObjectURL(file);

    uploadedVideo.src = url;

    uploadedVideo.style.display = 'block';

    uploadedImage.style.display = 'none';

    video.style.display = 'none';

    currentSource = uploadedVideo;

    uploadedVideo.onloadedmetadata = ()=>{

        uploadedVideo.play();

        resizeOverlay();
    };
});

function animationLoop(){

    if(faceDetected && lastFaceLocation){

        drawFaceFrame(lastFaceLocation,currentState);
    }

    requestAnimationFrame(animationLoop);
}

startCamera();

animationLoop();

setInterval(scanCycle,700);

setInterval(updateClock,1000);

updateClock();

hideBoxes();