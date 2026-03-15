from dataclasses import dataclass
import time

from toolbox.tasks.echo_task import EchoTask, Page
from toolbox.core.profile import EchoProfile
from toolbox.utils.ocr import detect_and_merge_rectangles_pil, ocr_pattern
from toolbox.utils.logger import logger

class EchoScan(EchoTask):
    """
    Scan all the echos in the main page and return the list of profiles.
    """
    def run(self) -> list[EchoProfile]:
        self.interaction.ensure_connected()
        logger.info("Scanning all echos in the main page")

        # 1. Ensure we are in the echo inspection page 
        self.to_page(Page.MAIN)

        for i in range(10):
            time.sleep(0.1)
            self.interaction.scroll(0.192, 0.244, -30)

        # 2. Scan all presented echo in the first page
        time.sleep(0.5)
        profiles = []
        width, height = self.interaction.get_app_window_size()

        left_top = (0.092, 0.231)
        right_bottom = (0.294, 0.835) 
        screenshot = self.interaction.screenshot_region(left_top[0], left_top[1], right_bottom[0], right_bottom[1])

        boxes = detect_and_merge_rectangles_pil(screenshot)

        for box in boxes:
            x, y, w, h = box
            self.interaction.click((x + w / 2) / width + left_top[0], (y + h / 2) / height + left_top[1])

            while True:
                profile_img = self.interaction.screenshot_region(0.7356, 0.1264, 0.952, 0.458)
                profile = EchoProfile().from_image(profile_img)

                if profile.validate():
                    if profile.level == 0:
                        # all following echos are not upgraded yet, skip the rest and return
                        return profiles
                        
                    profiles.append(profile)
                    break

                time.sleep(1)
            
        if len(profiles) < 15:
            # this indicates all echos have been scanned 
            logger.info("All echos have been scanned.")
            return profiles
        
        # 3. Scoll down to bottom line by line and scan each echo 
        last_line_valid = True
        continuous_valid_lines, continuous_invalid_lines = 0, 0

        self.interaction.scroll(0.192, 0.544, 8.0)
        while True:
            self.interaction.scroll(0.192, 0.544, 0.7)

            left_top = (0.087, 0.738)
            right_bottom = (0.297, 0.828)

            _tmp_screenshot = self.interaction.screenshot_region(left_top[0], left_top[1], right_bottom[0], right_bottom[1])
            boxes = detect_and_merge_rectangles_pil(_tmp_screenshot)

            if len(boxes) > 0:
                if last_line_valid is False:
                    # retake the screenshot to ensure the boxes are not missed
                    _tmp_screenshot = self.interaction.screenshot_region(left_top[0], left_top[1], 
                            right_bottom[0], right_bottom[1] + 0.1)
                    boxes = detect_and_merge_rectangles_pil(_tmp_screenshot)

                    for box in boxes:
                        x, y, w, h = box
                        self.interaction.click((x + w / 2) / width + left_top[0], (y + h / 2) / height + left_top[1])

                        # extract the echo profile 
                        while True:
                            profile_img = self.interaction.screenshot_region(0.7356, 0.1264, 0.952, 0.458)
                            # 显示图像
                            #profile_img.show()
                            profile = EchoProfile().from_image(profile_img)

                            if profile.validate():
                                if profile.level == 0:
                                    # all following echos are not upgraded yet, skip the rest and return
                                    return profiles

                                profiles.append(profile)
                                break

                            time.sleep(1)

                    self.interaction.scroll(0.192, 0.544, 4)

                last_line_valid = True
                continuous_valid_lines += 1
                continuous_invalid_lines = 0
            else:
                last_line_valid = False
                continuous_valid_lines = 0
                continuous_invalid_lines += 1
            
            # we reach the bottom of the whole page
            if continuous_valid_lines >= 20 or continuous_invalid_lines >= 20:
                break
        
        logger.info(f"Scanned {len(profiles)} echos")

        return profiles
        
