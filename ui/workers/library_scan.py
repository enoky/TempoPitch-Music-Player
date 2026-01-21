import os
from PySide6 import QtCore
from library import LibraryService
from metadata import probe_metadata

class LibraryScanWorker(QtCore.QObject):
    progress = QtCore.Signal(int, str)
    finished = QtCore.Signal(int, list) # count, added_paths
    preliminary_finished = QtCore.Signal() # Signal to indicate fast scan is done
    
    def __init__(self, library: LibraryService, paths: list[str], is_folder: bool, parent=None):
        super().__init__(parent)
        self._library = library
        self._paths = paths
        self._is_folder = is_folder
        self._abort = False

    @QtCore.Slot()
    def run(self) -> None:
        # Phase 1: Fast Scan (local files only)
        added_paths = []
        
        def fast_meta_extractor(path: str) -> dict:
            try:
                # We probing local only for speed in first pass
                m = probe_metadata(path, fetch_online=False)
                return {
                    "title": m.title,
                    "artist": m.artist,
                    "album": m.album,
                    "genre": m.genre,
                    "year": m.year,
                    "track_number": m.track_number,
                    "duration_sec": m.duration_sec,
                    "cover_art": m.cover_art,
                }
            except Exception:
                return {}

        def progress_wrapper(count: int, path: str):
            added_paths.append(path)
            self._emit_progress(count, path)

        count = 0
        if self._is_folder:
            for folder in self._paths:
                if self._abort: break
                count += self._library.scan_folder(
                    folder,
                    progress_callback=progress_wrapper,
                    metadata_extractor=fast_meta_extractor
                )
        else:
            count += self._library.scan_files(
                self._paths,
                progress_callback=progress_wrapper,
                metadata_extractor=fast_meta_extractor
            )
        
        # Notify that scanner is done
        self.preliminary_finished.emit()
        self.finished.emit(count, added_paths)

    def _emit_progress(self, count: int, path: str):
        self.progress.emit(count, f"Adding: {os.path.basename(path)}")

    def stop(self) -> None:
        self._abort = True
        self._library.abort_scan()


class MetadataWorker(QtCore.QObject):
    progress = QtCore.Signal(str) # status message
    finished = QtCore.Signal()
    
    def __init__(self, library: LibraryService, paths: list[str], parent=None):
        super().__init__(parent)
        self._library = library
        self._paths = paths
        self._abort = False

    @QtCore.Slot()
    def run(self) -> None:
        # Phase 2: Background Metadata Fetch
        if not self._paths:
            self.finished.emit()
            return

        self.progress.emit("Fetching online metadata...")
        for i, path in enumerate(self._paths):
            if self._abort: break
            
            try:
                m = probe_metadata(path, fetch_online=True)
                
                def full_meta_extractor(p: str) -> dict:
                    return {
                        "title": m.title,
                        "artist": m.artist,
                        "album": m.album,
                        "genre": m.genre,
                        "year": m.year,
                        "track_number": m.track_number,
                        "duration_sec": m.duration_sec,
                        "cover_art": m.cover_art,
                    }

                self._library.scan_files(
                    [path],
                    metadata_extractor=full_meta_extractor,
                    progress_callback=None 
                )
                
                if i % 5 == 0:
                    self.progress.emit(f"Updating metadata: {os.path.basename(path)}")
                    
            except Exception:
                pass

        self.finished.emit()

    def stop(self) -> None:
        self._abort = True
