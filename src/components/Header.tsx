
import { observer } from "mobx-react-lite";
import styles from "./styles/header.module.scss"


const Header = observer(() => {

  
    return (
      <header className={styles.header}>
        <div className={styles.title}>
          Переводчик
        </div>
      </header>
    );
  });
  
  export default Header;
  