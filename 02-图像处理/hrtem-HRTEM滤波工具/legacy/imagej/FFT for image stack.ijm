setBatchMode(true); 
    stack = getImageID; 
    newImage("FFT Movie of "+getTitle, "8-bit", getWidth, getHeight,   
nSlices); 
    movie = getImageID; 
    for (i=1; i<=nSlices; i++) { 
       showProgress(i, nSlices); 
       selectImage(stack); 
       setSlice(i); 
       run("FFT"); 
       run("Copy"); 
       close; 
       selectImage(movie); 
       setSlice(i); 
       run("Paste"); 
    } 
    setBatchMode(false);