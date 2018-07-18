(declaim (optimize (speed 3) (debug 0) (safety 0)))

(in-package :th)

(cffi:defcfun ("THFile_isOpened" th-file-is-opened) :int (self th-file-ptr))
(cffi:defcfun ("THFile_isQuiet" th-file-is-quiet) :int (self th-file-ptr))
(cffi:defcfun ("THFile_isReadable" th-file-is-readable) :int (self th-file-ptr))
(cffi:defcfun ("THFile_isWritable" th-file-is-writable) :int (self th-file-ptr))
(cffi:defcfun ("THFile_isBinary" th-file-is-binary) :int (self th-file-ptr))
(cffi:defcfun ("THFile_isAutoSpacing" th-file-is-auto-spacing) :int (self th-file-ptr))
(cffi:defcfun ("THFile_hasError" th-file-has-error) :int (self th-file-ptr))

(cffi:defcfun ("THFile_binary" th-file-binary) :void (self th-file-ptr))
(cffi:defcfun ("THFile_ascii" th-file-ascii) :void (self th-file-ptr))
(cffi:defcfun ("THFile_autoSpacing" th-file-auto-spacing) :void (self th-file-ptr))
(cffi:defcfun ("THFile_noAutoSpacing" th-file-no-auto-spacing) :void (self th-file-ptr))
(cffi:defcfun ("THFile_quiet" th-file-quiet) :void (self th-file-ptr))
(cffi:defcfun ("THFile_pedantic" th-file-pedantic) :void (self th-file-ptr))
(cffi:defcfun ("THFile_clearError" th-file-clear-error) :void (self th-file-ptr))

(cffi:defcfun ("THFile_readByteScalar" th-file-read-byte-scalar) :unsigned-char (self th-file-ptr))
(cffi:defcfun ("THFile_readCharScalar" th-file-read-char-scalar) :char (self th-file-ptr))
(cffi:defcfun ("THFile_readShortScalar" th-file-read-short-scalar) :short (self th-file-ptr))
(cffi:defcfun ("THFile_readIntScalar" th-file-read-int-scalar) :int (self th-file-ptr))
(cffi:defcfun ("THFile_readLongScalar" th-file-read-long-scalar) :long (self th-file-ptr))
(cffi:defcfun ("THFile_readFloatScalar" th-file-read-float-scalar) :float (self th-file-ptr))
(cffi:defcfun ("THFile_readDoubleScalar" th-file-read-double-scalar) :double (self th-file-ptr))

(cffi:defcfun ("THFile_writeByteScalar" th-file-write-byte-scalar) :long
  (self th-file-ptr)
  (scalar :unsigned-char))
(cffi:defcfun ("THFile_writeCharScalar" th-file-write-char-scalar) :long
  (self th-file-ptr)
  (scalar :char))
(cffi:defcfun ("THFile_writeShortScalar" th-file-write-short-scalar) :long
  (self th-file-ptr)
  (scalar :short))
(cffi:defcfun ("THFile_writeIntScalar" th-file-write-int-scalar) :long
  (self th-file-ptr)
  (scalar :int))
(cffi:defcfun ("THFile_writeLongScalar" th-file-write-long-scalar) :long
  (self th-file-ptr)
  (scalar :long))
(cffi:defcfun ("THFile_writeFloatScalar" th-file-write-float-scalar) :long
  (self th-file-ptr)
  (scalar :float))
(cffi:defcfun ("THFile_writeDoubleScalar" th-file-write-double-scalar) :long
  (self th-file-ptr)
  (scalar :double))

(cffi:defcfun ("THFile_readByte" th-file-read-byte) :long
  (self th-file-ptr)
  (storage th-byte-storage-ptr))
(cffi:defcfun ("THFile_readChar" th-file-read-char) :long
  (self th-file-ptr)
  (storage th-char-storage-ptr))
(cffi:defcfun ("THFile_readShort" th-file-read-short) :long
  (self th-file-ptr)
  (storage th-short-storage-ptr))
(cffi:defcfun ("THFile_readInt" th-file-read-int) :long
  (self th-file-ptr)
  (storage th-int-storage-ptr))
(cffi:defcfun ("THFile_readLong" th-file-read-long) :long
  (self th-file-ptr)
  (storage th-long-storage-ptr))
(cffi:defcfun ("THFile_readFloat" th-file-read-float) :long
  (self th-file-ptr)
  (storage th-float-storage-ptr))
(cffi:defcfun ("THFile_readDouble" th-file-read-double) :long
  (self th-file-ptr)
  (storage th-double-storage-ptr))

(cffi:defcfun ("THFile_writeByte" th-file-write-byte) :long
  (self th-file-ptr)
  (storage th-byte-storage-ptr))
(cffi:defcfun ("THFile_writeChar" th-file-write-char) :long
  (self th-file-ptr)
  (storage th-char-storage-ptr))
(cffi:defcfun ("THFile_writeShort" th-file-write-short) :long
  (self th-file-ptr)
  (storage th-short-storage-ptr))
(cffi:defcfun ("THFile_writeInt" th-file-write-int) :long
  (self th-file-ptr)
  (storage th-int-storage-ptr))
(cffi:defcfun ("THFile_writeLong" th-file-write-long) :long
  (self th-file-ptr)
  (storage th-long-storage-ptr))
(cffi:defcfun ("THFile_writeFloat" th-file-write-float) :long
  (self th-file-ptr)
  (storage th-float-storage-ptr))
(cffi:defcfun ("THFile_writeDouble" th-file-write-double) :long
  (self th-file-ptr)
  (storage th-double-storage-ptr))

(cffi:defcfun ("THFile_synchronize" th-file-synchronize) :void (self th-file-ptr))
(cffi:defcfun ("THFile_seek" th-file-seek) :void
  (self th-file-ptr)
  (position :long))
(cffi:defcfun ("THFile_seekEnd" th-file-seek-end) :void (self th-file-ptr))
(cffi:defcfun ("THFile_position" th-file-position) :long (self th-file-ptr))
(cffi:defcfun ("THFile_close" th-file-close) :void (self th-file-ptr))
(cffi:defcfun ("THFile_free" th-file-free) :void (self th-file-ptr))

(cffi:defcfun ("THDiskFile_new" th-disk-file-new) th-file-ptr
  (name :string)
  (mode :string)
  (quietp :int))
(cffi:defcfun ("THPipeFile_new" th-pipe-file-new) th-file-ptr
  (name :string)
  (mode :string)
  (quietp :int))

(cffi:defcfun ("THDiskFile_name" th-disk-file-name) :string (self th-file-ptr))

(cffi:defcfun ("THDiskFile_isLittleEndianCPU" th-disk-file-is-little-endian-cpu) :int)
(cffi:defcfun ("THDiskFile_isBigEndianCPU" th-disk-file-is-big-endian-cpu) :int)
(cffi:defcfun ("THDiskFile_nativeEndianEncoding" th-disk-file-native-endian-encoding) :void
  (self th-file-ptr))
(cffi:defcfun ("THDiskFile_littleEndianEncoding" th-disk-file-little-endian-encoding) :void
  (self th-file-ptr))
(cffi:defcfun ("THDiskFile_bigEndianEncoding" th-disk-file-big-endian-encoding) :void
  (self th-file-ptr))
(cffi:defcfun ("THDiskFile_longSize" th-disk-file-long-size) :void
  (self th-file-ptr)
  (size :int))
(cffi:defcfun ("THDiskFile_noBuffer" th-disk-file-no-buffer) :void (self th-file-ptr))

(cffi:defcfun ("THMemoryFile_newWithStorage" th-memory-file-new-with-storage) th-file-ptr
  (storage th-char-storage-ptr)
  (mode :string))
(cffi:defcfun ("THMemoryFile_new" th-memory-file-new) th-file-ptr (mode :string))

(cffi:defcfun ("THMemoryFile_storage" th-memory-file-storage) th-char-storage-ptr (self th-file-ptr))
(cffi:defcfun ("THMemoryFile_longSize" th-memory-file-long-size) :void
  (self th-file-ptr)
  (size :int))
