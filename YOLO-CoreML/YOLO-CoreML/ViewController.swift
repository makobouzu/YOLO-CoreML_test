import UIKit
import Vision
import AVFoundation

class ViewController: UIViewController {
  @IBOutlet weak var videoPreview: UIView!
  @IBOutlet weak var timeLabel: UILabel!
  @IBOutlet weak var personLabel: UILabel!
  @IBOutlet weak var laptopLabel: UILabel!
  @IBOutlet weak var bookLabel: UILabel!

  let yolo = YOLO()

  var videoCapture: VideoCapture!
  var request: VNCoreMLRequest!
  var startTimes: [CFTimeInterval] = []

  var boundingBoxes = [BoundingBox]()
  var colors: [UIColor] = []

  var framesDone = 0
  var frameCapturingStartTime = CACurrentMediaTime()
  let semaphore = DispatchSemaphore(value: 2)

  override func viewDidLoad() {
    super.viewDidLoad()

    timeLabel.text = ""

    setUpBoundingBoxes()
    setUpVision()
    setUpCamera()

    frameCapturingStartTime = CACurrentMediaTime()
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    print(#function)
  }

  // MARK: - Initialization

  func setUpBoundingBoxes() {
    for _ in 0..<YOLO.maxBoundingBoxes {
      boundingBoxes.append(BoundingBox())
    }
    for r: CGFloat in [0.2, 0.4, 0.6, 0.85, 1.0] {
      for g: CGFloat in [0.6, 0.7, 0.8, 0.9] {
        for b: CGFloat in [0.6, 0.7, 0.8, 1.0] {
          let color = UIColor(red: r, green: g, blue: b, alpha: 1)
          colors.append(color)
        }
      }
    }
  }
  
  func setUpVision() {
    guard let visionModel = try? VNCoreMLModel(for: yolo.model.model) else {
      print("Error: could not create Vision model")
      return
    }
    request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
    request.imageCropAndScaleOption = .scaleFit
  }

  func setUpCamera() {
    videoCapture = VideoCapture()
    videoCapture.delegate = self
    videoCapture.fps = 60
    videoCapture.setUp(sessionPreset: AVCaptureSession.Preset.vga640x480) { success in
      if success {
        // Add the video preview into the UI.
        if let previewLayer = self.videoCapture.previewLayer {
        //　動画の表示
          self.videoPreview.layer.addSublayer(previewLayer)
          self.resizePreviewLayer()
        }
        // Add the bounding box layers to the UI, on top of the video preview.
        for box in self.boundingBoxes {
          box.addToLayer(self.videoPreview.layer)
        }
        // Once everything is set up, we can start capturing live video.
        self.videoCapture.start()
      }
    }
  }

  // MARK: - UI stuff

  override func viewWillLayoutSubviews() {
    super.viewWillLayoutSubviews()
    resizePreviewLayer()
  }

  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }

//動画の描画サイズ(boundingboxとは別) - ここを変更
  func resizePreviewLayer() {
    videoCapture.previewLayer?.frame = CGRect(x: 0, y: 0, width: videoPreview.frame.width, height: videoPreview.frame.height)
  }

  // MARK: - Doing inference
  func predictUsingVision(pixelBuffer: CVPixelBuffer) {
    startTimes.append(CACurrentMediaTime())
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
    try? handler.perform([request])
  }

  func visionRequestDidComplete(request: VNRequest, error: Error?) {
    if let observations = request.results as? [VNCoreMLFeatureValueObservation],
       let features = observations.first?.featureValue.multiArrayValue {
      let boundingBoxes = yolo.computeBoundingBoxes(features: features)
      let elapsed = CACurrentMediaTime() - startTimes.remove(at: 0)
      showOnMainThread(boundingBoxes, elapsed)
    }
  }

  func showOnMainThread(_ boundingBoxes: [YOLO.Prediction], _ elapsed: CFTimeInterval) {
    DispatchQueue.main.async {
      self.show(predictions: boundingBoxes)

      let fps = self.measureFPS()
      self.timeLabel.text = String(format: "Elapsed %.5f seconds - %.2f FPS", elapsed, fps)
      self.semaphore.signal()
    }
  }

  func measureFPS() -> Double {
    // Measure how many frames were actually delivered per second.
    framesDone += 1
    let frameCapturingElapsed = CACurrentMediaTime() - frameCapturingStartTime
    let currentFPSDelivered = Double(framesDone) / frameCapturingElapsed
    if frameCapturingElapsed > 1 {
      framesDone = 0
      frameCapturingStartTime = CACurrentMediaTime()
    }
    return currentFPSDelivered
  }

  func show(predictions: [YOLO.Prediction]) {
    var personCount = 0
    var laptopCount = 0
    var bookCount   = 0
    
    for i in 0..<boundingBoxes.count {
      if i < predictions.count {
        let prediction = predictions[i]

        // The predicted bounding box is in the coordinate space of the input
        // image, which is a square image of 416x416 pixels. We want to show it
        // on the video preview, which is as wide as the screen and has a 4:3
        // aspect ratio. The video preview also may be letterboxed at the top
        // and bottom.
        let width = videoPreview.bounds.width
        let height = videoPreview.bounds.height
        let scaleX = width / CGFloat(YOLO.inputWidth)
        let scaleY = height / CGFloat(YOLO.inputHeight)
        let top = videoPreview.bounds.minY

        // Translate and scale the rectangle to our own coordinate system.
        var rect = prediction.rect
        rect.origin.x *= scaleX
        rect.origin.y *= scaleY
        rect.origin.y += top
        rect.size.width *= scaleX
        rect.size.height *= scaleY

        // Show the bounding box.
        let label = String(format: "%@ %.1f", labels[prediction.classIndex], prediction.score * 100)
        let color = colors[prediction.classIndex]
        boundingBoxes[i].show(frame: rect, label: label, color: color)
        if(labels[prediction.classIndex] == "person"){
            personCount += 1
        }
        if(labels[prediction.classIndex] == "laptop"){
            laptopCount += 1
        }
        if(labels[prediction.classIndex] == "book"){
            bookCount += 1
        }
      } else {
        boundingBoxes[i].hide()
        if i < predictions.count {
            let prediction = predictions[i]
            if(labels[prediction.classIndex] == "person"){
                personCount -= 1
            }
            if(labels[prediction.classIndex] == "laptop"){
                laptopCount -= 1
            }
            if(labels[prediction.classIndex] == "book"){
                bookCount -= 1
            }
        }
      }
        personLabel.text = ("\(personCount)人")
        laptopLabel.text = ("\(laptopCount)個")
        bookLabel.text   = ("\(bookCount)冊")
    }
  }
}

extension ViewController: VideoCaptureDelegate {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {
    semaphore.wait()

    if let pixelBuffer = pixelBuffer {
      
      DispatchQueue.global().async {
        self.predictUsingVision(pixelBuffer: pixelBuffer)
      }
    }
  }
}
