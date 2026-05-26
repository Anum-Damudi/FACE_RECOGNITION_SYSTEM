const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');

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

function resizeOverlay(){

    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;

    if(video.videoWidth && video.videoHeight){

        hiddenCanvas.width = video.videoWidth;
        hiddenCanvas.height = video.videoHeight;
    }
}

video.addEventListener('loadedmetadata',resizeOverlay);

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

    const pad = 35;

    const bx = x - pad;
    const by = y - pad;

    const bw = w + pad * 2;
    const bh = h + pad * 2;

    clearOverlay();

    let primary = '#00fff2';
    let glow = 'rgba(0,255,255,0.8)';

    if(state === 'known'){

        primary = '#00ff88';
        glow = 'rgba(0,255,136,0.8)';
    }

    if(state === 'unknown'){

        primary = '#ffd000';
        glow = 'rgba(255,208,0,0.8)';
    }

    if(state === 'criminal'){

        primary = '#ff2f2f';
        glow = 'rgba(255,47,47,0.8)';
    }

    ctx.save();

    // outer glow

    ctx.strokeStyle = primary;
    ctx.lineWidth = 3;

    ctx.shadowColor = glow;
    ctx.shadowBlur = 25;

    ctx.strokeRect(bx,by,bw,bh);

    // corners

    const c = 45;

    ctx.beginPath();

    ctx.moveTo(bx,by+c);
    ctx.lineTo(bx,by);
    ctx.lineTo(bx+c,by);

    ctx.moveTo(bx+bw-c,by);
    ctx.lineTo(bx+bw,by);
    ctx.lineTo(bx+bw,by+c);

    ctx.moveTo(bx,by+bh-c);
    ctx.lineTo(bx,by+bh);
    ctx.lineTo(bx+c,by+bh);

    ctx.moveTo(bx+bw-c,by+bh);
    ctx.lineTo(bx+bw,by+bh);
    ctx.lineTo(bx+bw,by+bh-c);

    ctx.stroke();

    // animated scan line

    const time = Date.now() / 8;

    const lineY = by + (time % bh);

    ctx.beginPath();

    ctx.strokeStyle = glow;
    ctx.lineWidth = 2;

    ctx.moveTo(bx + 5,lineY);
    ctx.lineTo(bx + bw - 5,lineY);

    ctx.stroke();

    // center target

    const cx = bx + bw / 2;
    const cy = by + bh / 2;

    ctx.beginPath();

    ctx.arc(cx,cy,8,0,Math.PI * 2);

    ctx.stroke();

    // crosshair

    ctx.beginPath();

    ctx.moveTo(cx - 20,cy);
    ctx.lineTo(cx + 20,cy);

    ctx.moveTo(cx,cy - 20);
    ctx.lineTo(cx,cy + 20);

    ctx.stroke();

    // grid

    ctx.lineWidth = 1;

    ctx.strokeStyle = 'rgba(0,255,255,0.18)';

    for(let i=1;i<4;i++){

        ctx.beginPath();

        ctx.moveTo(bx + (bw/4)*i,by);
        ctx.lineTo(bx + (bw/4)*i,by + bh);

        ctx.stroke();

        ctx.beginPath();

        ctx.moveTo(bx,by + (bh/4)*i);
        ctx.lineTo(bx + bw,by + (bh/4)*i);

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

    if(!video.videoWidth || !video.videoHeight){

        return null;
    }

    hiddenCanvas.width = video.videoWidth;
    hiddenCanvas.height = video.videoHeight;

    hiddenCtx.drawImage(
        video,
        0,
        0,
        hiddenCanvas.width,
        hiddenCanvas.height
    );

    return new Promise((resolve)=>{

        hiddenCanvas.toBlob((blob)=>{

            resolve(blob);

        },'image/jpeg',0.8);
    });
}

async function scanCycle(){

    if(processing || !video.videoWidth){

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

async function startCamera(){

    try{

        const stream = await navigator.mediaDevices.getUserMedia({

            video:{
                width:{ideal:1280},
                height:{ideal:720},
                facingMode:'user'
            },

            audio:false
        });

        video.srcObject = stream;

        await video.play();
    }

    catch(error){

        console.error('Camera Error:',error);

        setIdleState();
    }
}

function animationLoop(){

    if(faceDetected && lastFaceLocation){

        drawFaceFrame(lastFaceLocation,currentState);
    }

    requestAnimationFrame(animationLoop);
}

startCamera();

animationLoop();

setInterval(scanCycle,500);

setInterval(updateClock,1000);

updateClock();

hideBoxes();